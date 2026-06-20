/**
 * @file yolo_trt_node.hpp
 * @brief WL100 YOLOE-26 TensorRT C++ 实时检测 + D435i 深度距离估计节点
 *
 * 数据流：
 *   D435i RGB ──┐
 *               ├─ message_filters 同步 → 定时推理 → Detection2DArray
 *   D435i Depth ┘                                  └→ 标注 Image
 *
 * 性能目标：端到端延迟 < 20ms（对比 Python 版 ~46ms）
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <fstream>
#include <mutex>

// ROS 2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/string.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

// TensorRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace wl100_perception_cpp
{

/**
 * @brief TensorRT 日志器
 */
class TrtLogger : public nvinfer1::ILogger
{
public:
    explicit TrtLogger(rclcpp::Logger ros_logger)
        : ros_logger_(ros_logger) {}

    void log(Severity severity, const char* msg) noexcept override
    {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                RCLCPP_ERROR(ros_logger_, "[TRT] %s", msg);
                break;
            case Severity::kWARNING:
                RCLCPP_WARN(ros_logger_, "[TRT] %s", msg);
                break;
            case Severity::kINFO:
                RCLCPP_DEBUG(ros_logger_, "[TRT] %s", msg);
                break;
            default:
                break;
        }
    }

private:
    rclcpp::Logger ros_logger_;
};

/**
 * @brief 单个检测结果
 */
struct Detection
{
    float x1, y1, x2, y2;  // 原图坐标 bbox
    float confidence;
    int class_id;
    std::string class_name;
    float distance_m;       // 深度距离（米），NaN 表示无效
};

/**
 * @brief Letterbox 变换参数（用于坐标反算）
 */
struct LetterboxInfo
{
    float scale;    // 缩放比例
    float pad_x;    // x 方向 padding
    float pad_y;    // y 方向 padding
};

/**
 * @brief YOLOE-26 TensorRT C++ 推理节点
 */
class YoloTrtNode : public rclcpp::Node
{
public:
    explicit YoloTrtNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~YoloTrtNode() override;

private:
    // ── TensorRT 初始化 ──
    bool load_engine(const std::string& engine_path);
    bool allocate_buffers();

    // ── 推理管线 ──
    void synced_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg);
    void inference_tick();

    // ── 前处理 ──
    LetterboxInfo preprocess(const cv::Mat& img, float* gpu_input);

    // ── 后处理 ──
    std::vector<Detection> postprocess(
        const float* output0, int num_dets, int det_dim,
        const LetterboxInfo& lb, int orig_w, int orig_h);

    // ── 深度提取 ──
    float extract_depth(const cv::Mat& depth_cv,
                        int x1, int y1, int x2, int y2);

    // ── 标注图绘制 ──
    cv::Mat draw_annotations(const cv::Mat& image,
                             const std::vector<Detection>& detections);

    // ── 性能统计 ──
    void print_stats();

    // ── 参数 ──
    std::string engine_path_;
    std::vector<std::string> class_names_;
    float conf_threshold_;
    double infer_rate_;
    int input_size_;
    float depth_min_;
    float depth_max_;
    float roi_ratio_;
    double distance_lowpass_alpha_;
    double stats_interval_;
    std::vector<std::string> publish_classes_;  // 发布类别白名单

    // ── 相机内参 (用于 3D 反投影) ──
    float cam_fx_;
    float cam_fy_;
    float cam_cx_;
    float cam_cy_;

    // ── TensorRT 引擎 ──
    std::unique_ptr<TrtLogger> trt_logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // ── Zero-Copy 共享内存缓冲区 (Jetson CPU/GPU 共享 DRAM) ──
    // host_* : CPU 可读写的指针 (pinned mapped memory)
    // dev_*  : GPU 可读写的设备指针 (指向同一块物理内存)
    float* host_input_ = nullptr;          // [3 * 640 * 640]
    float* host_output0_ = nullptr;        // [300 * 38]
    float* host_output1_ = nullptr;        // [32 * 160 * 160] (暂不用)
    void* dev_input_ = nullptr;
    void* dev_output0_ = nullptr;
    void* dev_output1_ = nullptr;
    cudaStream_t cuda_stream_ = nullptr;

    // ── 引擎元数据 ──
    int num_classes_ = 14;
    int max_dets_ = 300;
    int det_dim_ = 38;      // 4 + 1 + 1 + 32
    int mask_dim_ = 32;

    // ── ROS 2 通信 ──
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr ann_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr vlm_trigger_pub_;

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::TimerBase::SharedPtr infer_timer_;
    rclcpp::TimerBase::SharedPtr stats_timer_;

    // ── 帧缓存 ──
    std::mutex frame_mutex_;
    sensor_msgs::msg::Image::ConstSharedPtr latest_rgb_;
    sensor_msgs::msg::Image::ConstSharedPtr latest_depth_;

    // ── 距离低通滤波缓存（按类）──
    std::mutex distance_mutex_;
    std::unordered_map<std::string, float> distance_ema_;

    // ── 性能统计 ──
    int infer_count_ = 0;
    double total_latency_ = 0.0;
    int total_detections_ = 0;
};

}  // namespace wl100_perception_cpp
