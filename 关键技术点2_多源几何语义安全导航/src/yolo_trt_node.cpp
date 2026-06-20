/**
 * @file yolo_trt_node.cpp
 * @brief WL100 YOLOE-26 TensorRT C++ 推理节点实现
 *
 * 核心优化点（对比 Python 版）:
 *   1. 直接调用 TensorRT C++ API，无 Python/ultralytics 开销
 *   2. GPU 缓冲区预分配，避免每帧 malloc/free
 *   3. CUDA Stream 异步执行
 *   4. NMS-free end2end 模型，输出直接解码
 *   5. OpenCV C++ 原生预处理
 */

#include "wl100_perception_cpp/yolo_trt_node.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <nlohmann/json.hpp>

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t status = call;                                          \
        if (status != cudaSuccess) {                                        \
            RCLCPP_FATAL(this->get_logger(), "CUDA Error: %s at %s:%d",     \
                         cudaGetErrorString(status), __FILE__, __LINE__);    \
            throw std::runtime_error("CUDA Error");                         \
        }                                                                   \
    } while (0)

namespace wl100_perception_cpp
{

// ═══════════════════════════════════════════════════
//  构造函数
// ═══════════════════════════════════════════════════
YoloTrtNode::YoloTrtNode(const rclcpp::NodeOptions& options)
    : Node("yolo_detector", options)
{
    // ── 参数声明 ──
    this->declare_parameter("model_path", "/home/nvidia/yoloe26-s-seg.engine");
    this->declare_parameter("class_names", std::vector<std::string>{
        "person", "chair", "table", "shelf", "door",
        "cardboard box", "trash can", "robot", "potted plant", "blackboard",
        "cable", "power strip", "electric cord", "plug"
    });
    this->declare_parameter("confidence_threshold", 0.4);
    this->declare_parameter("inference_rate", 5.0);
    this->declare_parameter("input_image_size", 640);
    this->declare_parameter("depth_min", 0.2);
    this->declare_parameter("depth_max", 8.0);
    this->declare_parameter("depth_roi_ratio", 0.5);
    this->declare_parameter("distance_lowpass_alpha", 0.3);
    this->declare_parameter("rgb_topic", "/camera/camera/color/image_raw");
    this->declare_parameter("depth_topic",
                            "/camera/camera/aligned_depth_to_color/image_raw");
    this->declare_parameter("detection_topic", "/yolo/detections");
    this->declare_parameter("annotated_image_topic", "/yolo/annotated_image");
    this->declare_parameter("marker_topic", "/yolo/markers");
    this->declare_parameter("stats_interval", 10.0);
    // 发布类别白名单（仅这些类别的检测结果会被发布，其余丢弃）
    this->declare_parameter("publish_classes", std::vector<std::string>{
        "chair", "cardboard box"
    });

    // 相机内参 (D435i 640×480 实际标定值，可通过 yaml 覆盖)
    // 注意: 每台 D435i 内参不同，建议从 camera_info 话题动态获取
    this->declare_parameter("cam_fx", 611.74);
    this->declare_parameter("cam_fy", 611.69);
    this->declare_parameter("cam_cx", 312.84);
    this->declare_parameter("cam_cy", 243.38);

    // ── 读取参数 ──
    engine_path_ = this->get_parameter("model_path").as_string();
    class_names_ = this->get_parameter("class_names").as_string_array();
    conf_threshold_ = this->get_parameter("confidence_threshold").as_double();
    infer_rate_ = this->get_parameter("inference_rate").as_double();
    input_size_ = this->get_parameter("input_image_size").as_int();
    depth_min_ = this->get_parameter("depth_min").as_double();
    depth_max_ = this->get_parameter("depth_max").as_double();
    roi_ratio_ = this->get_parameter("depth_roi_ratio").as_double();
    distance_lowpass_alpha_ = this->get_parameter("distance_lowpass_alpha").as_double();
    auto rgb_topic = this->get_parameter("rgb_topic").as_string();
    auto depth_topic = this->get_parameter("depth_topic").as_string();
    auto det_topic = this->get_parameter("detection_topic").as_string();
    auto ann_topic = this->get_parameter("annotated_image_topic").as_string();
    auto marker_topic = this->get_parameter("marker_topic").as_string();
    stats_interval_ = this->get_parameter("stats_interval").as_double();
    publish_classes_ = this->get_parameter("publish_classes").as_string_array();
    RCLCPP_INFO(this->get_logger(), "发布类别白名单: %zu 个类别", publish_classes_.size());

    cam_fx_ = static_cast<float>(this->get_parameter("cam_fx").as_double());
    cam_fy_ = static_cast<float>(this->get_parameter("cam_fy").as_double());
    cam_cx_ = static_cast<float>(this->get_parameter("cam_cx").as_double());
    cam_cy_ = static_cast<float>(this->get_parameter("cam_cy").as_double());

    num_classes_ = static_cast<int>(class_names_.size());

    // ── 加载 TensorRT 引擎 ──
    RCLCPP_INFO(this->get_logger(), "正在加载 TRT 引擎: %s ...", engine_path_.c_str());
    if (!load_engine(engine_path_)) {
        RCLCPP_FATAL(this->get_logger(), "TRT 引擎加载失败！");
        throw std::runtime_error("Engine load failed");
    }
    if (!allocate_buffers()) {
        RCLCPP_FATAL(this->get_logger(), "GPU 缓冲区分配失败！");
        throw std::runtime_error("Buffer allocation failed");
    }

    // 预热推理（GPU kernels 编译缓存）
    RCLCPP_INFO(this->get_logger(), "预热推理...");
    std::memset(host_input_, 0, 3 * input_size_ * input_size_ * sizeof(float));
    context_->enqueueV3(cuda_stream_);
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));

    RCLCPP_INFO(this->get_logger(),
                "TRT C++ 引擎加载完成，类别数: %d, 类别: [%s]",
                num_classes_, [&]() {
                    std::string s;
                    for (size_t i = 0; i < class_names_.size(); i++) {
                        if (i > 0) s += ", ";
                        s += class_names_[i];
                    }
                    return s;
                }().c_str());

    // ── 发布者 ──
    det_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
        det_topic, 10);
    ann_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        ann_topic, 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        marker_topic, 10);
    vlm_trigger_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/vlm/trigger", 10);

    // ── message_filters 同步订阅 ──
    rgb_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
        this, rgb_topic);
    depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
        this, depth_topic);
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(5), *rgb_sub_, *depth_sub_);
    sync_->registerCallback(&YoloTrtNode::synced_callback, this);
    RCLCPP_INFO(this->get_logger(), "已订阅 RGB: %s  Depth: %s",
                rgb_topic.c_str(), depth_topic.c_str());

    // ── 推理定时器 ──
    auto period_ms = std::chrono::milliseconds(
        static_cast<int>(1000.0 / infer_rate_));
    infer_timer_ = this->create_wall_timer(
        period_ms, std::bind(&YoloTrtNode::inference_tick, this));

    // ── 性能统计定时器 ──
    stats_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(stats_interval_),
        std::bind(&YoloTrtNode::print_stats, this));

    RCLCPP_INFO(this->get_logger(),
                "YoloTrtNode (C++) 已启动 (推理频率=%.1fHz, 置信度≥%.2f, "
                "深度范围=%.1f~%.1fm, box低通α=%.2f)",
                infer_rate_, conf_threshold_, depth_min_, depth_max_, distance_lowpass_alpha_);
}

// ═══════════════════════════════════════════════════
//  析构函数
// ═══════════════════════════════════════════════════
YoloTrtNode::~YoloTrtNode()
{
    if (cuda_stream_) {
        cudaStreamSynchronize(cuda_stream_);
        cudaStreamDestroy(cuda_stream_);
    }
    // Zero-Copy: 用 cudaFreeHost 释放映射内存
    if (host_input_) cudaFreeHost(host_input_);
    if (host_output0_) cudaFreeHost(host_output0_);
    if (host_output1_) cudaFreeHost(host_output1_);
    RCLCPP_INFO(this->get_logger(), "YoloTrtNode (C++) 正在退出...");
}

// ═══════════════════════════════════════════════════
//  加载 TRT 引擎
// ═══════════════════════════════════════════════════
bool YoloTrtNode::load_engine(const std::string& engine_path)
{
    // 读取引擎文件
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "无法打开引擎文件: %s", engine_path.c_str());
        return false;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // ultralytics 格式: 4 字节 metadata_size + JSON metadata + raw TRT engine
    uint32_t meta_size = 0;
    file.read(reinterpret_cast<char*>(&meta_size), sizeof(meta_size));

    // 读取并解析 metadata
    std::vector<char> meta_buf(meta_size);
    file.read(meta_buf.data(), meta_size);
    std::string meta_json(meta_buf.begin(), meta_buf.end());

    try {
        auto meta = nlohmann::json::parse(meta_json);
        RCLCPP_INFO(this->get_logger(), "引擎 metadata: task=%s, imgsz=%s, batch=%d",
                     meta.value("task", "unknown").c_str(),
                     meta.value("imgsz", nlohmann::json::array()).dump().c_str(),
                     meta.value("batch", 1));

        // 从 metadata 更新类名
        if (meta.contains("names")) {
            class_names_.clear();
            auto& names = meta["names"];
            for (auto it = names.begin(); it != names.end(); ++it) {
                int idx = std::stoi(it.key());
                if (idx >= static_cast<int>(class_names_.size())) {
                    class_names_.resize(idx + 1);
                }
                class_names_[idx] = it.value().get<std::string>();
            }
            num_classes_ = static_cast<int>(class_names_.size());
        }
    } catch (const std::exception& e) {
        RCLCPP_WARN(this->get_logger(), "metadata 解析失败: %s，使用参数中的类名", e.what());
    }

    // 读取纯 TRT 引擎数据
    size_t trt_size = file_size - 4 - meta_size;
    std::vector<char> engine_data(trt_size);
    file.read(engine_data.data(), trt_size);
    file.close();

    RCLCPP_INFO(this->get_logger(), "TRT 引擎大小: %.1f MB", trt_size / 1024.0 / 1024.0);

    // 创建 TensorRT runtime & engine
    trt_logger_ = std::make_unique<TrtLogger>(this->get_logger());
    runtime_.reset(nvinfer1::createInferRuntime(*trt_logger_));
    if (!runtime_) {
        RCLCPP_ERROR(this->get_logger(), "创建 TRT Runtime 失败");
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), trt_size));
    if (!engine_) {
        RCLCPP_ERROR(this->get_logger(), "反序列化 TRT 引擎失败");
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        RCLCPP_ERROR(this->get_logger(), "创建执行上下文失败");
        return false;
    }

    // 打印 IO 绑定信息
    for (int i = 0; i < engine_->getNbIOTensors(); i++) {
        auto name = engine_->getIOTensorName(i);
        auto shape = engine_->getTensorShape(name);
        auto mode = engine_->getTensorIOMode(name);
        std::string shape_str = "[";
        for (int d = 0; d < shape.nbDims; d++) {
            if (d > 0) shape_str += ", ";
            shape_str += std::to_string(shape.d[d]);
        }
        shape_str += "]";
        RCLCPP_INFO(this->get_logger(), "  IO[%d] %s: shape=%s, %s",
                     i, name, shape_str.c_str(),
                     mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT");
    }

    // 解析输出维度
    auto out0_shape = engine_->getTensorShape("output0");
    if (out0_shape.nbDims == 3) {
        max_dets_ = out0_shape.d[1];
        det_dim_ = out0_shape.d[2];
    }
    RCLCPP_INFO(this->get_logger(), "max_dets=%d, det_dim=%d, num_classes=%d",
                max_dets_, det_dim_, num_classes_);

    return true;
}

// ═══════════════════════════════════════════════════
//  分配 GPU 缓冲区
// ═══════════════════════════════════════════════════
bool YoloTrtNode::allocate_buffers()
{
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));

    // ── Zero-Copy: cudaHostAllocMapped 分配 CPU/GPU 共享内存 ──
    // Jetson Orin NX 的 CPU 和 GPU 共享同一块 DRAM，
    // 使用 Mapped Pinned Memory 可以完全消除 cudaMemcpy H2D/D2H

    // 输入: [1, 3, 640, 640]
    size_t input_bytes = 1 * 3 * input_size_ * input_size_ * sizeof(float);
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_input_),
               input_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_input_, host_input_, 0));

    // 输出0: [1, 300, 38]
    size_t output0_bytes = 1 * max_dets_ * det_dim_ * sizeof(float);
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_output0_),
               output0_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_output0_, host_output0_, 0));

    // 输出1: [1, 32, 160, 160] (mask prototypes, 暂不用但需要分配)
    size_t output1_bytes = 1 * 32 * 160 * 160 * sizeof(float);
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_output1_),
               output1_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&dev_output1_, host_output1_, 0));

    // 绑定张量地址（使用 GPU 端指针）
    context_->setTensorAddress("images", dev_input_);
    context_->setTensorAddress("output0", dev_output0_);
    context_->setTensorAddress("output1", dev_output1_);

    RCLCPP_INFO(this->get_logger(),
                "Zero-Copy 缓冲区分配完成: input=%.1fMB, output0=%.1fKB, output1=%.1fMB",
                input_bytes / 1024.0 / 1024.0,
                output0_bytes / 1024.0,
                output1_bytes / 1024.0 / 1024.0);

    return true;
}

// ═══════════════════════════════════════════════════
//  同步回调：缓存最新帧
// ═══════════════════════════════════════════════════
void YoloTrtNode::synced_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
{
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_rgb_ = rgb_msg;
    latest_depth_ = depth_msg;
}

// ═══════════════════════════════════════════════════
//  定时器回调：执行推理
// ═══════════════════════════════════════════════════
void YoloTrtNode::inference_tick()
{
    sensor_msgs::msg::Image::ConstSharedPtr rgb_msg, depth_msg;
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (!latest_rgb_ || !latest_depth_) return;
        rgb_msg = latest_rgb_;
        depth_msg = latest_depth_;
    }

    auto t_start = std::chrono::steady_clock::now();

    // ── 图像转换 ──
    cv_bridge::CvImageConstPtr rgb_cv_ptr, depth_cv_ptr;
    try {
        rgb_cv_ptr = cv_bridge::toCvShare(rgb_msg, "bgr8");
        depth_cv_ptr = cv_bridge::toCvShare(depth_msg);
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_WARN(this->get_logger(), "图像转换失败: %s", e.what());
        return;
    }

    const cv::Mat& rgb_cv = rgb_cv_ptr->image;
    const cv::Mat& depth_cv = depth_cv_ptr->image;

    // ── 前处理（直接写入 Zero-Copy 共享内存）──
    LetterboxInfo lb = preprocess(rgb_cv, nullptr);

    // Zero-Copy: 无需 cudaMemcpy，GPU 直接读取 host_input_
    // 执行推理
    context_->enqueueV3(cuda_stream_);
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));

    // Zero-Copy: 无需 cudaMemcpy D2H，直接从 host_output0_ 读取
    // ── 后处理 ──
    auto detections = postprocess(
        host_output0_, max_dets_, det_dim_,
        lb, rgb_cv.cols, rgb_cv.rows);

    // ── 深度提取 ──
    for (auto& det : detections) {
        det.distance_m = extract_depth(depth_cv,
            static_cast<int>(det.x1), static_cast<int>(det.y1),
            static_cast<int>(det.x2), static_cast<int>(det.y2));
    }

    // ── 类别白名单过滤 ──
    if (!publish_classes_.empty()) {
        detections.erase(
            std::remove_if(detections.begin(), detections.end(),
                [this](const Detection& d) {
                    return std::find(publish_classes_.begin(),
                                     publish_classes_.end(),
                                     d.class_name) == publish_classes_.end();
                }),
            detections.end());
    }

    // ── 距离低通滤波（白名单所有类别）──
    for (auto& det : detections) {
        if (std::find(publish_classes_.begin(), publish_classes_.end(),
                      det.class_name) == publish_classes_.end()) {
            continue;
        }

        std::lock_guard<std::mutex> lock(distance_mutex_);
        auto it = distance_ema_.find(det.class_name);

        if (std::isnan(det.distance_m)) {
            if (it != distance_ema_.end()) {
                det.distance_m = it->second;
            }
            continue;
        }

        float filtered = det.distance_m;
        if (it == distance_ema_.end()) {
            distance_ema_[det.class_name] = filtered;
        } else {
            filtered = static_cast<float>(
                distance_lowpass_alpha_ * det.distance_m +
                (1.0 - distance_lowpass_alpha_) * it->second);
            it->second = filtered;
        }
        det.distance_m = filtered;
    }

    // ── 构建 Detection2DArray ──
    auto det_array = vision_msgs::msg::Detection2DArray();
    det_array.header = rgb_msg->header;

    for (const auto& det : detections) {
        vision_msgs::msg::Detection2D det_msg;
        det_msg.header = rgb_msg->header;

        // bbox (中心 + 尺寸)
        float cx = (det.x1 + det.x2) / 2.0f;
        float cy = (det.y1 + det.y2) / 2.0f;
        float w = det.x2 - det.x1;
        float h = det.y2 - det.y1;
        det_msg.bbox.center.position.x = cx;
        det_msg.bbox.center.position.y = cy;
        det_msg.bbox.size_x = w;
        det_msg.bbox.size_y = h;

        // 类别 + 置信度
        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = det.class_name;
        hyp.hypothesis.score = det.confidence;
        det_msg.results.push_back(hyp);

        // 距离信息
        if (!std::isnan(det.distance_m)) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s %.2fm", det.class_name.c_str(), det.distance_m);
            det_msg.id = buf;
        } else {
            det_msg.id = det.class_name + " --m";
        }

        det_array.detections.push_back(det_msg);
    }

    det_pub_->publish(det_array);

    // ── VLM 触发: 有检测结果时发布 /vlm/trigger (所有物体) ──
    if (vlm_trigger_pub_->get_subscription_count() > 0 && !detections.empty()) {
        // 构建所有检测结果的 JSON 数组
        nlohmann::json det_array_json = nlohmann::json::array();
        for (const auto& det : detections) {
            nlohmann::json det_json;
            det_json["class"] = det.class_name;
            det_json["confidence"] = std::round(det.confidence * 100.0) / 100.0;
            if (!std::isnan(det.distance_m)) {
                det_json["distance"] = std::round(det.distance_m * 100.0) / 100.0;
            } else {
                det_json["distance"] = nullptr;
            }
            det_array_json.push_back(det_json);
        }
        std_msgs::msg::String trigger_msg;
        trigger_msg.data = det_array_json.dump();
        vlm_trigger_pub_->publish(trigger_msg);
    }

    // ── 发布 3D 文字 Marker (RViz2 可视化) ──
    if (marker_pub_->get_subscription_count() > 0 && !detections.empty()) {
        visualization_msgs::msg::MarkerArray marker_array;
        int marker_id = 0;

        for (const auto& det : detections) {
            if (std::isnan(det.distance_m) || det.distance_m <= 0.0f) {
                continue;  // 无深度的不标注
            }

            // bbox 中心像素坐标
            float u = (det.x1 + det.x2) / 2.0f;
            float v = (det.y1 + det.y2) / 2.0f;
            float Z = det.distance_m;

            // Pinhole 反投影: 像素 → camera_color_optical_frame 3D 坐标
            float X = (u - cam_cx_) * Z / cam_fx_;
            float Y = (v - cam_cy_) * Z / cam_fy_;

            visualization_msgs::msg::Marker marker;
            marker.header = rgb_msg->header;  // frame_id = camera_color_optical_frame
            marker.ns = "yolo_labels";
            marker.id = marker_id++;
            marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            marker.action = visualization_msgs::msg::Marker::ADD;

            // 文字位置 (略高于物体中心)
            marker.pose.position.x = X;
            marker.pose.position.y = Y - 0.15;  // 光学坐标系 Y 朝下，减小 Y 使文字偏上
            marker.pose.position.z = Z;
            marker.pose.orientation.w = 1.0;

            // 字号: 三轴统一设置 (RViz2 Bug #1336: 只设 z 会导致字距异常)
            marker.scale.x = 0.08;
            marker.scale.y = 0.08;
            marker.scale.z = 0.08;

            // 白色文字
            marker.color.r = 1.0f;
            marker.color.g = 1.0f;
            marker.color.b = 1.0f;
            marker.color.a = 1.0f;

            // 自动过期 (略大于推理周期，防闪烁)
            marker.lifetime = rclcpp::Duration::from_seconds(0.3);

            // 文字内容: 类别 + 距离
            char text_buf[64];
            snprintf(text_buf, sizeof(text_buf), "%s %.1fm",
                     det.class_name.c_str(), det.distance_m);
            marker.text = text_buf;

            marker_array.markers.push_back(marker);
        }

        if (!marker_array.markers.empty()) {
            marker_pub_->publish(marker_array);
        }
    }

    // ── 发布标注图像 ──
    if (ann_pub_->get_subscription_count() > 0) {
        auto ann_cv = draw_annotations(rgb_cv, detections);
        auto ann_msg = cv_bridge::CvImage(
            rgb_msg->header, "bgr8", ann_cv).toImageMsg();
        ann_pub_->publish(*ann_msg);
    }

    // ── 性能统计 ──
    auto t_end = std::chrono::steady_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    infer_count_++;
    total_latency_ += latency_ms;
    total_detections_ += static_cast<int>(detections.size());
}

// ═══════════════════════════════════════════════════
//  Letterbox 前处理
// ═══════════════════════════════════════════════════
LetterboxInfo YoloTrtNode::preprocess(const cv::Mat& img, float* /*gpu_input*/)
{
    int orig_w = img.cols;
    int orig_h = img.rows;
    int target = input_size_;

    // 计算 letterbox 缩放
    float scale = std::min(
        static_cast<float>(target) / orig_w,
        static_cast<float>(target) / orig_h);
    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);
    float pad_x = (target - new_w) / 2.0f;
    float pad_y = (target - new_h) / 2.0f;

    // Resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Padding (灰色填充 114)
    cv::Mat padded(target, target, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(
        static_cast<int>(pad_x), static_cast<int>(pad_y), new_w, new_h)));

    // BGR → RGB + Normalize [0, 1] + HWC → CHW
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    // HWC → CHW (交错存储到 Zero-Copy 共享内存 host_input_)
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    size_t channel_size = target * target;
    for (int c = 0; c < 3; c++) {
        std::memcpy(host_input_ + c * channel_size,
                    channels[c].data,
                    channel_size * sizeof(float));
    }

    return {scale, pad_x, pad_y};
}

// ═══════════════════════════════════════════════════
//  后处理（NMS-free end2end）
// ═══════════════════════════════════════════════════
std::vector<Detection> YoloTrtNode::postprocess(
    const float* output0, int num_dets, int det_dim,
    const LetterboxInfo& lb, int orig_w, int orig_h)
{
    std::vector<Detection> results;
    results.reserve(32);

    // output0 shape: [300, 38]
    // 每个检测: [x1, y1, x2, y2, conf, cls_id, mask_coeffs...]
    for (int i = 0; i < num_dets; i++) {
        const float* det = output0 + i * det_dim;

        float conf = det[4];
        if (conf < conf_threshold_) continue;

        // bbox 在 640x640 letterboxed 空间
        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        int cls_id = static_cast<int>(det[5]);

        // 反算回原图坐标
        x1 = (x1 - lb.pad_x) / lb.scale;
        y1 = (y1 - lb.pad_y) / lb.scale;
        x2 = (x2 - lb.pad_x) / lb.scale;
        y2 = (y2 - lb.pad_y) / lb.scale;

        // Clamp 到原图范围
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

        if (x2 <= x1 || y2 <= y1) continue;

        Detection d;
        d.x1 = x1;
        d.y1 = y1;
        d.x2 = x2;
        d.y2 = y2;
        d.confidence = conf;
        d.class_id = cls_id;
        d.class_name = (cls_id >= 0 && cls_id < num_classes_)
                       ? class_names_[cls_id] : "class_" + std::to_string(cls_id);
        d.distance_m = std::numeric_limits<float>::quiet_NaN();

        results.push_back(d);
    }

    return results;
}

// ═══════════════════════════════════════════════════
//  深度距离提取
// ═══════════════════════════════════════════════════
float YoloTrtNode::extract_depth(
    const cv::Mat& depth_cv, int x1, int y1, int x2, int y2)
{
    int h = depth_cv.rows;
    int w = depth_cv.cols;

    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(w, x2);
    y2 = std::min(h, y2);

    if (x2 <= x1 || y2 <= y1) return std::numeric_limits<float>::quiet_NaN();

    // 取中心 ROI
    int bw = x2 - x1;
    int bh = y2 - y1;
    int margin_x = static_cast<int>(bw * (1.0f - roi_ratio_) / 2.0f);
    int margin_y = static_cast<int>(bh * (1.0f - roi_ratio_) / 2.0f);
    int roi_x1 = x1 + margin_x;
    int roi_y1 = y1 + margin_y;
    int roi_x2 = x2 - margin_x;
    int roi_y2 = y2 - margin_y;

    if (roi_x2 <= roi_x1 || roi_y2 <= roi_y1) {
        roi_x1 = x1; roi_y1 = y1; roi_x2 = x2; roi_y2 = y2;
    }

    cv::Mat roi = depth_cv(cv::Rect(roi_x1, roi_y1,
                                     roi_x2 - roi_x1, roi_y2 - roi_y1));

    // 收集有效深度值
    std::vector<float> valid;
    valid.reserve(roi.total());

    if (roi.type() == CV_16UC1) {
        for (int r = 0; r < roi.rows; r++) {
            const uint16_t* ptr = roi.ptr<uint16_t>(r);
            for (int c = 0; c < roi.cols; c++) {
                if (ptr[c] > 0) {
                    valid.push_back(static_cast<float>(ptr[c]));
                }
            }
        }
    } else if (roi.type() == CV_32FC1) {
        for (int r = 0; r < roi.rows; r++) {
            const float* ptr = roi.ptr<float>(r);
            for (int c = 0; c < roi.cols; c++) {
                if (ptr[c] > 0) {
                    valid.push_back(ptr[c] * 1000.0f);  // m → mm
                }
            }
        }
    }

    if (valid.empty()) return std::numeric_limits<float>::quiet_NaN();

    // 中值
    size_t mid = valid.size() / 2;
    std::nth_element(valid.begin(), valid.begin() + mid, valid.end());
    float depth_m = valid[mid] / 1000.0f;  // mm → m

    if (depth_m < depth_min_ || depth_m > depth_max_) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    return depth_m;
}

// ═══════════════════════════════════════════════════
//  标注图绘制
// ═══════════════════════════════════════════════════
cv::Mat YoloTrtNode::draw_annotations(
    const cv::Mat& image, const std::vector<Detection>& detections)
{
    cv::Mat ann = image.clone();

    for (const auto& det : detections) {
        int x1 = static_cast<int>(det.x1);
        int y1 = static_cast<int>(det.y1);
        int x2 = static_cast<int>(det.x2);
        int y2 = static_cast<int>(det.y2);

        bool has_dist = !std::isnan(det.distance_m);
        cv::Scalar color = has_dist ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);

        cv::rectangle(ann, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        char text[128];
        if (has_dist) {
            snprintf(text, sizeof(text), "%s %.2fm (%.0f%%)",
                     det.class_name.c_str(), det.distance_m, det.confidence * 100);
        } else {
            snprintf(text, sizeof(text), "%s --m (%.0f%%)",
                     det.class_name.c_str(), det.confidence * 100);
        }

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                              0.5, 1, &baseline);
        cv::rectangle(ann, cv::Point(x1, y1 - text_size.height - 6),
                      cv::Point(x1 + text_size.width, y1), color, -1);
        cv::putText(ann, text, cv::Point(x1, y1 - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1,
                    cv::LINE_AA);
    }

    return ann;
}

// ═══════════════════════════════════════════════════
//  性能统计
// ═══════════════════════════════════════════════════
void YoloTrtNode::print_stats()
{
    if (infer_count_ == 0) {
        if (!latest_rgb_) {
            RCLCPP_WARN(this->get_logger(),
                        "尚未收到相机数据，请检查 D435i 是否启动");
        }
        return;
    }

    double avg_lat = total_latency_ / infer_count_;
    double avg_det = static_cast<double>(total_detections_) / infer_count_;
    RCLCPP_INFO(this->get_logger(),
                "[性能] 推理=%d帧, 平均延迟=%.1fms, "
                "平均检测数=%.1f, 实际帧率=%.1fHz",
                infer_count_, avg_lat, avg_det,
                infer_count_ / stats_interval_);

    infer_count_ = 0;
    total_latency_ = 0.0;
    total_detections_ = 0;
}

}  // namespace wl100_perception_cpp

// ═══════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<wl100_perception_cpp::YoloTrtNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("yolo_trt"), "节点异常退出: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
