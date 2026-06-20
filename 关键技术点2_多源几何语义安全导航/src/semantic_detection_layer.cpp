#include "wl100_semantic_costmap/semantic_detection_layer.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "pluginlib/class_list_macros.hpp"
#include "nav2_costmap_2d/costmap_math.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

PLUGINLIB_EXPORT_CLASS(
    wl100_semantic_costmap::SemanticDetectionLayer,
    nav2_costmap_2d::Layer)

namespace wl100_semantic_costmap
{

// ═══════════════════════════════════════════════════
//  初始化
// ═══════════════════════════════════════════════════
void SemanticDetectionLayer::onInitialize()
{
    auto node = node_.lock();
    if (!node) {
        throw std::runtime_error("SemanticDetectionLayer: 无法获取 node 引用");
    }

    // ── 声明参数 ──
    declareParameter("enabled", rclcpp::ParameterValue(true));
    declareParameter("detection_topic", rclcpp::ParameterValue("/yolo/detections"));
    declareParameter("decay_time", rclcpp::ParameterValue(2.0));
    declareParameter("min_confidence", rclcpp::ParameterValue(0.40));
    declareParameter("transform_tolerance", rclcpp::ParameterValue(0.3));
    declareParameter("max_detection_range", rclcpp::ParameterValue(3.5));

    // D435i 内参 (640×480 实际标定值)
    declareParameter("fx", rclcpp::ParameterValue(611.74));
    declareParameter("fy", rclcpp::ParameterValue(611.69));
    declareParameter("cx", rclcpp::ParameterValue(312.84));
    declareParameter("cy", rclcpp::ParameterValue(243.38));

    // ── 读取参数 ──
    node->get_parameter(name_ + ".enabled", enabled_);
    node->get_parameter(name_ + ".detection_topic", detection_topic_);
    node->get_parameter(name_ + ".decay_time", decay_time_);
    node->get_parameter(name_ + ".min_confidence", min_confidence_);
    node->get_parameter(name_ + ".transform_tolerance", transform_tolerance_);
    node->get_parameter(name_ + ".max_detection_range", max_detection_range_);
    node->get_parameter(name_ + ".fx", fx_);
    node->get_parameter(name_ + ".fy", fy_);
    node->get_parameter(name_ + ".cx", cx_);
    node->get_parameter(name_ + ".cy", cy_);

    // ── 加载类别代价映射 ──
    loadClassCosts();

    // ── 订阅检测话题 ──
    detection_sub_ = node->create_subscription<vision_msgs::msg::Detection2DArray>(
        detection_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(&SemanticDetectionLayer::detectionCallback, this, std::placeholders::_1));

    // ── VLM override 订阅 ──
    declareParameter("override_topic", rclcpp::ParameterValue("/vlm/inflate_override"));
    declareParameter("vlm_tau_base", rclcpp::ParameterValue(20.0));
    declareParameter("tau_beta", rclcpp::ParameterValue(0.5));
    declareParameter("tau_min", rclcpp::ParameterValue(3.0));
    declareParameter("tau_max", rclcpp::ParameterValue(20.0));
    declareParameter("psi_min", rclcpp::ParameterValue(-6.0));
    declareParameter("psi_max", rclcpp::ParameterValue(6.0));
    declareParameter("gamma", rclcpp::ParameterValue(1.0));
    declareParameter("log_lr_pos", rclcpp::ParameterValue(0.8));
    declareParameter("log_lr_neg", rclcpp::ParameterValue(0.8));
    declareParameter("yolo_stale_timeout", rclcpp::ParameterValue(1.0));
    declareParameter("distance_consistency_scale", rclcpp::ParameterValue(0.8));
    declareParameter("conf_min", rclcpp::ParameterValue(0.4));
    node->get_parameter(name_ + ".override_topic", override_topic_);
    node->get_parameter(name_ + ".vlm_tau_base", vlm_tau_base_);
    node->get_parameter(name_ + ".tau_beta", tau_beta_);
    node->get_parameter(name_ + ".tau_min", tau_min_);
    node->get_parameter(name_ + ".tau_max", tau_max_);
    node->get_parameter(name_ + ".psi_min", psi_min_);
    node->get_parameter(name_ + ".psi_max", psi_max_);
    node->get_parameter(name_ + ".gamma", gamma_);
    node->get_parameter(name_ + ".log_lr_pos", log_lr_pos_);
    node->get_parameter(name_ + ".log_lr_neg", log_lr_neg_);
    node->get_parameter(name_ + ".yolo_stale_timeout", yolo_stale_timeout_);
    node->get_parameter(name_ + ".distance_consistency_scale", distance_consistency_scale_);
    node->get_parameter(name_ + ".conf_min", conf_min_);

    override_sub_ = node->create_subscription<std_msgs::msg::String>(
        override_topic_,
        10,
        std::bind(&SemanticDetectionLayer::overrideCallback, this, std::placeholders::_1));

    // ── DRSS 状态发布器（用于 rosbag 录制 CMPT 时序数据）──
    drss_state_pub_ = node->create_publisher<std_msgs::msg::String>(
        "/drss/state", rclcpp::SensorDataQoS());

    // 标记层为 "current" — 否则 costmap 被判定为"过期"，MPPI 控制器会 abort
    current_ = true;

    RCLCPP_INFO(node->get_logger(),
        "SemanticDetectionLayer 初始化完成: topic=%s, override=%s, decay=%.1fs, "
        "vlm_tau=%.1fs, tau_beta=%.1f, tau=[%.1f, %.1f], psi=[%.1f, %.1f], "
        "min_conf=%.2f, max_range=%.1fm, 类别数=%zu",
        detection_topic_.c_str(), override_topic_.c_str(),
        decay_time_, vlm_tau_base_, tau_beta_, tau_min_, tau_max_, psi_min_, psi_max_,
        min_confidence_,
        max_detection_range_, class_costs_.size());
}

// ═══════════════════════════════════════════════════
//  加载类别代价配置
// ═══════════════════════════════════════════════════
void SemanticDetectionLayer::loadClassCosts()
{
    // 默认类别代价映射（覆盖 14 类 YOLOE-26 检测目标）
    // cost: 代价值 (0=忽略, 50=低, 150=中, 200=高, 254=致命)
    // radius: 物体占据半径 (m)，InflationLayer 会在此基础上再膨胀
    // decay_time: 该类别障碍物保持时间 (s)
    //   动态物体 (person/robot) = 3s → 离开视野后快速清除幽灵
    //   静态物体 = 20s → 近距离盲区保护

    // ── 安全关键：致命代价（动态物体，快速衰减）──
    class_costs_["person"]            = {254, 0.20,  3.0};   // 行人
    class_costs_["robot"]             = {254, 0.25,  3.0};   // 其他机器人

    // ── 地面障碍物：致命代价（静态）──
    class_costs_["cable"]             = {254, 0.08, 20.0};   // 线缆
    class_costs_["power strip"]       = {254, 0.08, 20.0};   // 插线板
    class_costs_["electric cord"]     = {254, 0.08, 20.0};   // 电源线
    class_costs_["plug"]              = {254, 0.05, 20.0};   // 插头

    // ── 常见障碍物：致命代价（静态）──
    class_costs_["chair"]             = {254, 0.15, 20.0};   // 椅子
    class_costs_["table"]             = {254, 0.25, 20.0};   // 桌子
    class_costs_["shelf"]             = {254, 0.25, 20.0};   // 货架
    class_costs_["cardboard box"]     = {254, 0.15, 20.0};   // 纸箱
    class_costs_["trash can"]         = {254, 0.10, 20.0};   // 垃圾桶
    class_costs_["potted plant"]      = {254, 0.10, 20.0};   // 盆栽

    // ── 环境元素：低/忽略（静态）──
    class_costs_["door"]              = {50,  0.08, 20.0};   // 门
    class_costs_["blackboard"]        = {0,   0.00, 20.0};   // 黑板
}

// ═══════════════════════════════════════════════════
//  检测回调
// ═══════════════════════════════════════════════════
void SemanticDetectionLayer::detectionCallback(
    const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
    if (!enabled_) return;

    std::vector<SemanticObstacle> new_obstacles;
    std::unordered_map<std::string, ClassEvidence> frame_evidence;
    int dynamic_count = 0;  // 统计动态物体数 (person+robot)
    rclcpp::Time frame_stamp(msg->header.stamp);

    for (const auto & det : msg->detections) {
        if (det.results.empty()) continue;

        // ── 解析字段 ──
        // YOLO 节点:
        //   hypothesis.class_id = "shelf"           (纯类名)
        //   hypothesis.score    = 0.44              (置信度)
        //   det.id              = "shelf 5.08m"     (类名+距离) 或 "shelf --m"
        const auto & hyp = det.results[0].hypothesis;
        std::string class_name = hyp.class_id;
        double confidence = hyp.score;

        // 置信度过滤
        if (confidence < min_confidence_) continue;

        // 统计动态物体（置信度过滤后、深度过滤前，确保视野中有人即被统计）
        if (class_name == "person" || class_name == "robot") {
            ++dynamic_count;
        }

        // 从 det.id 字段解析距离
        double distance_m = 0.0;
        bool has_depth = false;
        const std::string & id_str = det.id;

        if (!id_str.empty() && id_str.find("--m") == std::string::npos) {
            // 格式: "shelf 5.08m" → 提取最后一个空格后的数字
            size_t space_pos = id_str.rfind(' ');
            if (space_pos != std::string::npos) {
                std::string dist_str = id_str.substr(space_pos + 1);
                if (!dist_str.empty() && dist_str.back() == 'm') {
                    try {
                        distance_m = std::stod(dist_str.substr(0, dist_str.size() - 1));
                        has_depth = (distance_m > 0.1 && distance_m < max_detection_range_);
                    } catch (...) {
                        has_depth = false;
                    }
                }
            }
        }

        // 无深度或超范围则跳过
        if (!has_depth) continue;

        // 构建当前帧类级证据摘要（仅统计有有效深度的证据，避免远处/无深度同类误证实）
        auto & evidence = frame_evidence[class_name];
        evidence.stamp = frame_stamp;
        evidence.present = true;
        evidence.count += 1;
        evidence.best_conf = std::max(evidence.best_conf, confidence);
        evidence.min_distance = std::min(evidence.min_distance, distance_m);

        // 查表确认是否需要标记
        auto it = class_costs_.find(class_name);
        if (it == class_costs_.end() || it->second.cost == 0) continue;

        // bbox 中心像素坐标
        double u = det.bbox.center.position.x;
        double v = det.bbox.center.position.y;

        // 3D 投影到 odom 坐标
        double ox, oy;
        if (projectToOdom(u, v, distance_m, msg->header.frame_id,
                          rclcpp::Time(msg->header.stamp), ox, oy))
        {
            SemanticObstacle obs;
            obs.x = ox;
            obs.y = oy;
            obs.class_name = class_name;
            obs.confidence = confidence;
            obs.distance = distance_m;
            obs.bbox_width = det.bbox.size_x;
            obs.stamp = rclcpp::Time(msg->header.stamp);
            new_obstacles.push_back(obs);

            auto node = node_.lock();
            if (node) {
                RCLCPP_DEBUG_THROTTLE(node->get_logger(), *node->get_clock(), 5000,
                    "[语义层] %s (%.0f%%) dist=%.2fm → odom(%.2f, %.2f) bbox=(%.0f,%.0f)",
                    class_name.c_str(), confidence * 100,
                    distance_m, ox, oy, u, v);
            }
        } else {
            auto node = node_.lock();
            if (node) {
                RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 2000,
                    "[语义层] TF失败: %s dist=%.2fm frame=%s",
                    class_name.c_str(), distance_m, msg->header.frame_id.c_str());
            }
        }
    }

    // 线程安全更新缓存
    if (!new_obstacles.empty()) {
        std::lock_guard<std::mutex> lock(obstacles_mutex_);

        // 添加新检测（不删除旧的，由衰减机制清理）
        for (auto & obs : new_obstacles) {
            // 去重：如果已有同类+近距离的障碍物，更新而不是追加
            bool merged = false;
            for (auto & existing : obstacles_) {
                double dx = existing.x - obs.x;
                double dy = existing.y - obs.y;
                double dist = std::hypot(dx, dy);
                if (existing.class_name == obs.class_name && dist < 0.5) {
                    // 更新位置和时间戳
                    existing.x = obs.x;
                    existing.y = obs.y;
                    existing.confidence = obs.confidence;
                    existing.distance = obs.distance;
                    existing.bbox_width = obs.bbox_width;
                    existing.stamp = obs.stamp;
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                obstacles_.push_back(obs);
            }
        }
        has_new_data_ = true;

        // P_vis: map 坐标已在 overrideCallback 中初始化，此处不再覆盖
        // 避免绕行时侧面深度不准或误检同类物体导致坐标跳变
    }

    // 更新动态物体计数（原子操作，无锁跨线程安全）
    n_dynamic_.store(dynamic_count, std::memory_order_relaxed);

    // 更新 YOLO 最新证据快照（放在 obstacles 更新之后，避免与 updateCosts 形成锁顺序反转）
    {
        std::lock_guard<std::mutex> evidence_lock(evidence_mutex_);
        latest_class_evidence_ = std::move(frame_evidence);
        latest_yolo_frame_stamp_ = frame_stamp;
    }
}

// ═══════════════════════════════════════════════════
//  VLM override 回调
// ═══════════════════════════════════════════════════
void SemanticDetectionLayer::overrideCallback(
    const std_msgs::msg::String::SharedPtr msg)
{
    auto node = node_.lock();
    if (!node) return;

    try {
        auto j = nlohmann::json::parse(msg->data);

        std::string class_name = j.value("class", "");
        double radius = j.value("radius", 0.3);
        double risk = j.value("risk_level", 0.5);
        std::string action = j.value("action", "NORMAL");

        if (class_name.empty()) return;

        // P_vis: 预读 obstacles_ 坐标（仅持 obstacles_mutex_，不嵌套）
        double pre_map_x = 0.0;
        double pre_map_y = 0.0;
        if (action != "NORMAL") {
            std::lock_guard<std::mutex> obs_lock(obstacles_mutex_);
            double best_dist_sq = 1e18;
            for (const auto & obs : obstacles_) {
                if (obs.class_name == class_name) {
                    double d2 = obs.x * obs.x + obs.y * obs.y;
                    if (d2 < best_dist_sq) {
                        pre_map_x = obs.x;
                        pre_map_y = obs.y;
                        best_dist_sq = d2;
                    }
                }
            }
        }

        std::lock_guard<std::mutex> lock(override_mutex_);

        if (action == "NORMAL") {
            if (radius_overrides_.erase(class_name) > 0) {
                RCLCPP_INFO(node->get_logger(),
                    "VLM override 清除: class=%s → 恢复默认值",
                    class_name.c_str());
            }
        } else {
            auto it = class_costs_.find(class_name);
            double default_radius = (it != class_costs_.end()) ? it->second.radius : 0.3;
            double final_radius = std::max(default_radius, radius);
            double claimed_distance = -1.0;

            if (j.contains("distance") && j["distance"].is_number()) {
                claimed_distance = j["distance"].get<double>();
            }

            RadiusOverride ov;
            ov.radius = final_radius;
            ov.cost = riskToCost(risk);
            ov.neg_log_lambda = 0.0;
            ov.last_update_time = node->get_clock()->now();
            ov.psi_score = 0.0;
            ov.last_evidence_stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
            ov.claimed_distance = claimed_distance;
            ov.miss_count = 0;
            ov.map_x = pre_map_x;
            ov.map_y = pre_map_y;

            bool is_new = (radius_overrides_.find(class_name) == radius_overrides_.end());
            radius_overrides_[class_name] = ov;

            RCLCPP_INFO(node->get_logger(),
                "VLM override %s: class=%s, radius=%.2fm, dist=%.2fm, cost=%u, risk=%.2f, τ=%.1fs",
                is_new ? "新增" : "更新",
                class_name.c_str(), final_radius, claimed_distance,
                static_cast<unsigned int>(ov.cost), risk, vlm_tau_base_);
        }
    } catch (const nlohmann::json::exception& e) {
        RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 5000,
            "VLM override JSON 解析失败: %s", e.what());
    }
}

// ═══════════════════════════════════════════════════
//  risk_level → cost 映射 (Language-as-Cost)
// ═══════════════════════════════════════════════════
unsigned char SemanticDetectionLayer::riskToCost(double risk)
{
    // risk < 0.3       → 200 (低危: MPPI 可穿越但尽量避)
    // 0.3 ≤ risk < 0.6 → 240 (中危: 强烈避开)
    // risk ≥ 0.6       → 254 (高危: LETHAL 不可穿越)
    if (risk < 0.3) return 200;
    if (risk < 0.6) return 240;
    return 254;
}

// ═══════════════════════════════════════════════════
//  3D 投影：像素 + 深度 → odom 坐标
// ═══════════════════════════════════════════════════
bool SemanticDetectionLayer::projectToOdom(
    double u, double v, double depth_m,
    const std::string & source_frame,
    const rclcpp::Time & stamp,
    double & out_x, double & out_y)
{
    // 相机坐标系 (optical frame: Z 前, X 右, Y 下)
    geometry_msgs::msg::PointStamped cam_point;
    cam_point.header.frame_id = source_frame;
    // 使用 Time(0) 获取最新可用 TF，避免 camera 时间戳领先 HDL TF 导致
    // extrapolation into future 错误 (~25ms 偏差对导航精度无影响)
    cam_point.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
    cam_point.point.x = (u - cx_) * depth_m / fx_;
    cam_point.point.y = (v - cy_) * depth_m / fy_;
    cam_point.point.z = depth_m;

    try {
        // 变换到 costmap 的全局坐标系 (通常是 odom)
        std::string target_frame = layered_costmap_->getGlobalFrameID();
        auto tf_point = tf_->transform(
            cam_point, target_frame,
            tf2::durationFromSec(transform_tolerance_));
        out_x = tf_point.point.x;
        out_y = tf_point.point.y;
        return true;
    } catch (const tf2::TransformException & ex) {
        auto node = node_.lock();
        if (node) {
            RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 5000,
                "SemanticDetectionLayer TF 变换失败: %s", ex.what());
        }
        return false;
    }
}

// ═══════════════════════════════════════════════════
//  P_vis：可见性先验计算
//  将障碍物 map 坐标变换到 base_link 坐标系，
//  根据水平角度和距离计算"如果障碍物还在原位，传感器应该看到它的概率"
//  返回 [0, 1]：1 = 在 FOV 正中心（强负证据），0 = 完全在视野外（不扣分）
// ═══════════════════════════════════════════════════
double SemanticDetectionLayer::computeVisibilityPrior(
    double obs_map_x, double obs_map_y)
{
    // 坐标为零说明从未初始化过（VLM 创建时 obstacles_ 里没有同类）
    if (std::abs(obs_map_x) < 1e-6 && std::abs(obs_map_y) < 1e-6) {
        return 1.0;  // 安全默认：视为可见，允许正常扣分
    }

    try {
        // 将障碍物的 odom 坐标变换到 base_link
        geometry_msgs::msg::PointStamped odom_point;
        std::string global_frame = layered_costmap_->getGlobalFrameID();  // 通常是 "odom"
        odom_point.header.frame_id = global_frame;
        odom_point.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);  // 最新 TF
        odom_point.point.x = obs_map_x;
        odom_point.point.y = obs_map_y;
        odom_point.point.z = 0.0;

        auto base_point = tf_->transform(
            odom_point, "base_link",
            tf2::durationFromSec(transform_tolerance_));

        double dx = base_point.point.x;  // base_link: X 前, Y 左
        double dy = base_point.point.y;

        // 距离
        double dist = std::hypot(dx, dy);

        // 距离因子：超出检测范围则 P_vis 降低
        double p_range = 1.0;
        double range_fade_start = max_detection_range_;        // 3.5m
        double range_fade_end = max_detection_range_ + 1.5;    // 5.0m
        if (dist > range_fade_end) {
            p_range = 0.0;
        } else if (dist > range_fade_start) {
            p_range = 1.0 - (dist - range_fade_start) / (range_fade_end - range_fade_start);
        }

        // 在后方（dx <= 0）→ 完全不可见
        if (dx <= 0.0) {
            return 0.0;
        }

        // 水平角度（相对于 base_link 正前方）
        double angle = std::abs(std::atan2(dy, dx));

        // 角度因子：在 FOV 内 → 1.0，边缘线性过渡，FOV 外 → 0.0
        double p_angle = 1.0;
        if (angle > sensor_half_hfov_ + pvis_fade_width_) {
            p_angle = 0.0;  // 完全在 FOV 外
        } else if (angle > sensor_half_hfov_) {
            // 过渡带：线性衰减
            p_angle = 1.0 - (angle - sensor_half_hfov_) / pvis_fade_width_;
        }

        double p_vis = p_angle * p_range;

        auto node_ptr = node_.lock();
        if (node_ptr) {
            RCLCPP_DEBUG_THROTTLE(node_ptr->get_logger(), *node_ptr->get_clock(), 3000,
                "[P_vis] obs_odom=(%.2f,%.2f) → base_link=(%.2f,%.2f) "
                "angle=%.1f° dist=%.2fm p_angle=%.3f p_range=%.3f → P_vis=%.3f",
                obs_map_x, obs_map_y, dx, dy,
                angle * 180.0 / M_PI, dist, p_angle, p_range, p_vis);
        }

        return std::clamp(p_vis, 0.0, 1.0);

    } catch (const tf2::TransformException & ex) {
        auto node_ptr = node_.lock();
        if (node_ptr) {
            RCLCPP_WARN_THROTTLE(node_ptr->get_logger(), *node_ptr->get_clock(), 5000,
                "[P_vis] TF 变换失败 (odom→base_link): %s → 默认 P_vis=1.0", ex.what());
        }
        return 1.0;  // TF 失败时安全默认：允许正常扣分
    }
}

// ═══════════════════════════════════════════════════
//  updateBounds：计算需要更新的区域
// ═══════════════════════════════════════════════════
void SemanticDetectionLayer::updateBounds(
    double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
    double * min_x, double * min_y,
    double * max_x, double * max_y)
{
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(obstacles_mutex_);

    auto node = node_.lock();
    if (!node) return;
    rclcpp::Time now = node->get_clock()->now();

    // CMPT 权限交接：取活跃 override 类名快照（override 还在的，不清理 obstacles_）
    std::unordered_set<std::string> active_override_classes;
    {
        std::lock_guard<std::mutex> olock(override_mutex_);
        for (const auto & [cls, ov] : radius_overrides_) {
            if (std::exp(-ov.neg_log_lambda) > 0.01) {
                active_override_classes.insert(cls);
            }
        }
    }

    // 清理过期障碍物（有 active override 的不清理 → 保证 ③ 一直有坐标可画）
    obstacles_.erase(
        std::remove_if(obstacles_.begin(), obstacles_.end(),
            [&](const SemanticObstacle & obs) {
                if (active_override_classes.count(obs.class_name) > 0) return false;
                double dt = decay_time_;  // 默认处理
                auto it = class_costs_.find(obs.class_name);
                if (it != class_costs_.end()) {
                    dt = it->second.decay_time;
                }
                return (now - obs.stamp).seconds() > dt;
            }),
        obstacles_.end());

    // 扩展 bounds（使用与 updateCosts 一致的动态半径）
    for (const auto & obs : obstacles_) {
        auto it = class_costs_.find(obs.class_name);
        double radius = (it != class_costs_.end()) ? it->second.radius : 0.3;

        // 检查是否有 VLM override（只读积分值，不步进 — 步进仅在 updateCosts 执行）
        {
            std::lock_guard<std::mutex> olock(override_mutex_);
            auto ov_it = radius_overrides_.find(obs.class_name);
            if (ov_it != radius_overrides_.end()) {
                double lambda = std::exp(-ov_it->second.neg_log_lambda);  // P2 保证严格单调
                if (lambda > 0.01) {
                    double r_eff = radius + lambda * (ov_it->second.radius - radius);
                    radius = std::max(radius, r_eff);
                }
            }
        }

        // 动态 bbox 半径（与 updateCosts 一致）
        double real_radius = radius;
        if (obs.bbox_width > 0.0 && obs.distance > 0.1) {
            real_radius = (obs.bbox_width * obs.distance) / (2.0 * fx_);
        }
        real_radius = std::max(real_radius, radius);

        *min_x = std::min(*min_x, obs.x - real_radius);
        *min_y = std::min(*min_y, obs.y - real_radius);
        *max_x = std::max(*max_x, obs.x + real_radius);
        *max_y = std::max(*max_y, obs.y + real_radius);
    }

    // 将上一轮语义层写入过的区域继续并入本轮 bounds，
    // 让 LayeredCostmap 自己去 reset + 重绘，避免直接修改 master_grid。
    if (!marked_cells_.empty() && layered_costmap_ && layered_costmap_->getCostmap()) {
        auto * costmap = layered_costmap_->getCostmap();
        const unsigned int size_x = costmap->getSizeInCellsX();
        const double resolution = costmap->getResolution();

        unsigned int min_mx = size_x;
        unsigned int min_my = costmap->getSizeInCellsY();
        unsigned int max_mx = 0;
        unsigned int max_my = 0;
        bool have_marked_bounds = false;

        for (unsigned int idx : marked_cells_) {
            unsigned int mx = idx % size_x;
            unsigned int my = idx / size_x;
            if (mx >= size_x || my >= costmap->getSizeInCellsY()) continue;
            min_mx = std::min(min_mx, mx);
            min_my = std::min(min_my, my);
            max_mx = std::max(max_mx, mx);
            max_my = std::max(max_my, my);
            have_marked_bounds = true;
        }

        if (have_marked_bounds) {
            double marked_min_x, marked_min_y, marked_max_x, marked_max_y;
            costmap->mapToWorld(min_mx, min_my, marked_min_x, marked_min_y);
            costmap->mapToWorld(max_mx, max_my, marked_max_x, marked_max_y);
            *min_x = std::min(*min_x, marked_min_x - 0.5 * resolution);
            *min_y = std::min(*min_y, marked_min_y - 0.5 * resolution);
            *max_x = std::max(*max_x, marked_max_x + 0.5 * resolution);
            *max_y = std::max(*max_y, marked_max_y + 0.5 * resolution);
        }
    }

    // ── CMPT 权限交接：override 还活着但 obstacles_ 已过期的，用 override 坐标扩展 bounds ──
    {
        std::lock_guard<std::mutex> olock(override_mutex_);
        std::unordered_set<std::string> obs_classes;
        for (const auto & obs : obstacles_) {
            obs_classes.insert(obs.class_name);
        }
        for (const auto & [cls, ov] : radius_overrides_) {
            if (obs_classes.count(cls) > 0) continue;  // 已在 obstacles_ 中处理
            if (std::abs(ov.map_x) < 1e-6 && std::abs(ov.map_y) < 1e-6) continue;
            double lambda = std::exp(-ov.neg_log_lambda);
            if (lambda <= 0.01) continue;
            auto cost_it = class_costs_.find(cls);
            double r_def = (cost_it != class_costs_.end()) ? cost_it->second.radius : 0.15;
            double r_eff = std::max(r_def, r_def + lambda * (ov.radius - r_def));
            *min_x = std::min(*min_x, ov.map_x - r_eff);
            *min_y = std::min(*min_y, ov.map_y - r_eff);
            *max_x = std::max(*max_x, ov.map_x + r_eff);
            *max_y = std::max(*max_y, ov.map_y + r_eff);
        }
    }
}

// ═══════════════════════════════════════════════════
//  updateCosts：在 costmap 格栅上写入代价
// ═══════════════════════════════════════════════════
void SemanticDetectionLayer::updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j)
{
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(obstacles_mutex_);

    // 上一轮标记区域已在 updateBounds() 中并入 bounds，由 LayeredCostmap 统一 reset。
    marked_cells_.clear();

    std::unordered_map<std::string, ClassEvidence> evidence_snapshot;
    rclcpp::Time evidence_frame_stamp{0, 0, RCL_ROS_TIME};
    {
        std::lock_guard<std::mutex> evidence_lock(evidence_mutex_);
        evidence_snapshot = latest_class_evidence_;
        evidence_frame_stamp = latest_yolo_frame_stamp_;
    }

    // ② 统一步进：每个 costmap update 周期对全部 override 积分一次
    //    目的：保证"每周期仅步进一次"，与论文 λ(t)=exp(-∫ds/τ(s)) 表述严格一致
    //    不在障碍物循环里步进，避免同类 obs 重复积分的语义歧义
    {
        std::lock_guard<std::mutex> olock(override_mutex_);
        auto node_ptr = node_.lock();
        rclcpp::Time step_now = node_ptr ? node_ptr->get_clock()->now() : rclcpp::Time(0, 0, RCL_ROS_TIME);
        int nd = n_dynamic_.load(std::memory_order_relaxed);

        std::vector<std::string> expired_keys;
        for (auto & [cls, ov] : radius_overrides_) {
            if (evidence_frame_stamp > ov.last_evidence_stamp) {
                auto ev_it = evidence_snapshot.find(cls);
                if (ev_it != evidence_snapshot.end() && ev_it->second.present) {
                    double w_conf = std::clamp(
                        (ev_it->second.best_conf - conf_min_) / std::max(1e-6, 1.0 - conf_min_),
                        0.0, 1.0);
                    double w_dist = 1.0;
                    if (ov.claimed_distance > 0.0 && ev_it->second.min_distance < 1e8) {
                        double dist_err = std::abs(ev_it->second.min_distance - ov.claimed_distance);
                        w_dist = std::exp(-dist_err / std::max(1e-6, distance_consistency_scale_));
                    }
                    ov.psi_score += log_lr_pos_ * w_conf * w_dist;
                    ov.miss_count = 0;
                } else {
                    ov.miss_count += 1;
                    if (ov.miss_count >= 3) {
                        // P_vis: 可见性先验加权负证据
                        double p_vis = computeVisibilityPrior(ov.map_x, ov.map_y);
                        ov.psi_score -= log_lr_neg_ * p_vis;
                        ov.miss_count = 0;
                        if (node_ptr) {
                            RCLCPP_INFO_THROTTLE(node_ptr->get_logger(), *node_ptr->get_clock(), 2000,
                                "[P_vis] class=%s P_vis=%.3f ψ扣分=%.3f (lr_neg=%.2f) → ψ=%.2f  "
                                "obs_map=(%.2f,%.2f)",
                                cls.c_str(), p_vis, log_lr_neg_ * p_vis, log_lr_neg_,
                                ov.psi_score, ov.map_x, ov.map_y);
                        }
                    }
                }
                ov.psi_score = std::clamp(ov.psi_score, psi_min_, psi_max_);
                ov.last_evidence_stamp = evidence_frame_stamp;
            }

            bool yolo_stale = (
                evidence_frame_stamp.nanoseconds() == 0 ||
                (step_now - evidence_frame_stamp).seconds() > yolo_stale_timeout_);
            double trust = 0.0;
            if (!yolo_stale) {
                trust = 1.0 / (1.0 + std::exp(-gamma_ * ov.psi_score));
            }

            double tau_scene = vlm_tau_base_ / (1.0 + tau_beta_ * static_cast<double>(nd));
            double tau_scene_clamped = std::clamp(tau_scene, tau_min_, tau_max_);
            double tau_eff = tau_min_ + (tau_scene_clamped - tau_min_) * trust;
            tau_eff = std::clamp(tau_eff, tau_min_, tau_max_);

            double dt_step = (step_now - ov.last_update_time).seconds();
            bool capped = (dt_step > 1.0);
            dt_step = std::max(0.0, std::min(dt_step, 1.0));  // 上限 1s，防 Nav2 卡顿大步跳变

            if (capped && node_ptr) {
                RCLCPP_WARN_THROTTLE(node_ptr->get_logger(), *node_ptr->get_clock(), 10000,
                    "[DRSS] step cap 触发: class=%s dt_raw>1s → 截断为 1s (Nav2 调度延迟?)",
                    cls.c_str());
            }

            ov.neg_log_lambda += dt_step / tau_eff;
            ov.last_update_time = step_now;

            // 提前收集已衰减完毕的 key（λ<0.01 ↔ ∫>ln(100)≈4.605）
            if (ov.neg_log_lambda > 4.605) {
                expired_keys.push_back(cls);
                if (node_ptr) {
                    RCLCPP_INFO(node_ptr->get_logger(),
                        "[DRSS] override 衰减完毕: class=%s ∫=%.2f ψ=%.2f trust=%.2f τ_eff=%.1fs "
                        "(τ_scene=%.1f,N=%d,β=%.1f) → 恢复默认",
                        cls.c_str(), ov.neg_log_lambda, ov.psi_score, trust, tau_eff,
                        tau_scene_clamped, nd, tau_beta_);
                }
            }
        }
        for (const auto & k : expired_keys) {
            radius_overrides_.erase(k);
            // VLM override 消退 → 同步清除同类 YOLO 语义障碍物
            // 注：obstacles_mutex_ 已在 updateCosts 入口处持有，此处可安全操作
            obstacles_.erase(
                std::remove_if(obstacles_.begin(), obstacles_.end(),
                    [&k](const SemanticObstacle & obs) {
                        return obs.class_name == k;
                    }),
                obstacles_.end());
        }

        // ── 发布 DRSS 状态到 /drss/state（用于 rosbag 录制）──
        if (drss_state_pub_ && node_ptr) {
            nlohmann::json state_json;
            state_json["stamp"] = step_now.seconds();
            state_json["n_dynamic"] = nd;
            state_json["n_obstacles"] = static_cast<int>(obstacles_.size());

            nlohmann::json overrides_arr = nlohmann::json::array();
            for (const auto & [cls, ov] : radius_overrides_) {
                double lambda_val = std::exp(-ov.neg_log_lambda);
                bool ys = (evidence_frame_stamp.nanoseconds() == 0 ||
                           (step_now - evidence_frame_stamp).seconds() > yolo_stale_timeout_);
                double tr = ys ? 0.0 : 1.0 / (1.0 + std::exp(-gamma_ * ov.psi_score));
                double ts = vlm_tau_base_ / (1.0 + tau_beta_ * static_cast<double>(nd));
                double tsc = std::clamp(ts, tau_min_, tau_max_);
                double te = std::clamp(tau_min_ + (tsc - tau_min_) * tr, tau_min_, tau_max_);

                // 计算当前 r_eff
                auto cost_it = class_costs_.find(cls);
                double r_def = (cost_it != class_costs_.end()) ? cost_it->second.radius : 0.15;
                double r_eff = r_def + lambda_val * (ov.radius - r_def);
                r_eff = std::max(r_def, r_eff);

                nlohmann::json ov_json;
                ov_json["class"] = cls;
                ov_json["lambda"] = std::round(lambda_val * 1000.0) / 1000.0;
                ov_json["psi"] = std::round(ov.psi_score * 100.0) / 100.0;
                ov_json["trust"] = std::round(tr * 1000.0) / 1000.0;
                ov_json["tau_eff"] = std::round(te * 10.0) / 10.0;
                ov_json["r_eff"] = std::round(r_eff * 1000.0) / 1000.0;
                ov_json["r_default"] = std::round(r_def * 1000.0) / 1000.0;
                ov_json["neg_log_lambda"] = std::round(ov.neg_log_lambda * 1000.0) / 1000.0;
                ov_json["miss_count"] = ov.miss_count;
                ov_json["claimed_distance"] = std::round(ov.claimed_distance * 100.0) / 100.0;
                // P_vis: 在 DRSS 状态中发布可见性先验，用于 rosbag 分析
                double pv = computeVisibilityPrior(ov.map_x, ov.map_y);
                ov_json["p_vis"] = std::round(pv * 1000.0) / 1000.0;
                ov_json["obs_map_x"] = std::round(ov.map_x * 100.0) / 100.0;
                ov_json["obs_map_y"] = std::round(ov.map_y * 100.0) / 100.0;
                overrides_arr.push_back(ov_json);
            }
            state_json["overrides"] = overrides_arr;

            std_msgs::msg::String state_msg;
            state_msg.data = state_json.dump();
            drss_state_pub_->publish(state_msg);
        }
    }

    // ③ 标记当前存在的语义障碍物（只读 lambda，不再步进）
    for (const auto & obs : obstacles_) {
        auto it = class_costs_.find(obs.class_name);
        if (it == class_costs_.end()) continue;

        unsigned char cost = it->second.cost;
        double radius = it->second.radius;

        // 只读 VLM override（积分已在步骤 ② 完成，P2 严格单调性由统一步进保证）
        {
            std::lock_guard<std::mutex> olock(override_mutex_);
            auto ov_it = radius_overrides_.find(obs.class_name);
            if (ov_it != radius_overrides_.end()) {
                double lambda = std::exp(-ov_it->second.neg_log_lambda);
                if (lambda > 0.01) {
                    // 安全性质 P1: r_eff ≥ r_default
                    double r_eff = radius + lambda * (ov_it->second.radius - radius);
                    radius = std::max(radius, r_eff);
                    cost = std::max(cost, ov_it->second.cost);  // 安全单调性：不降低 cost
                    auto node_ptr = node_.lock();
                    if (node_ptr) {
                        RCLCPP_INFO_THROTTLE(node_ptr->get_logger(), *node_ptr->get_clock(), 5000,
                            "[DRSS] class=%s λ=%.3f ∫=%.3f ψ=%.2f dist_claim=%.2f "
                            "r_eff=%.3fm r_default=%.3fm",
                            obs.class_name.c_str(), lambda, ov_it->second.neg_log_lambda,
                            ov_it->second.psi_score, ov_it->second.claimed_distance,
                            r_eff, it->second.radius);
                    }
                }
                // lambda ≤ 0.01 的清除已在统一步进块里处理，此处无需再 erase
            }
        }


        // 世界坐标 → 格栅坐标
        unsigned int mx, my;
        if (!master_grid.worldToMap(obs.x, obs.y, mx, my)) continue;

        // 根据 bbox 像素宽度 + 距离动态计算物体真实半径
        // real_radius = bbox_width_pixels * distance / (2 * fx)
        double default_radius = it->second.radius;  // YOLO 默认半径（核心区）
        double real_radius = radius;  // VLM 衰减后的半径（可能更大）
        if (obs.bbox_width > 0.0 && obs.distance > 0.1) {
            double bbox_radius = (obs.bbox_width * obs.distance) / (2.0 * fx_);
            default_radius = std::max(default_radius, bbox_radius);
            real_radius = std::max(real_radius, bbox_radius);
        }
        real_radius = std::max(real_radius, radius);

        int cell_radius = static_cast<int>(real_radius / master_grid.getResolution());
        if (cell_radius < 1) cell_radius = 1;  // 最小 1 格保证可见


        int imx = static_cast<int>(mx);
        int imy = static_cast<int>(my);



        for (int dy = -cell_radius; dy <= cell_radius; dy++) {
            for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                int dist_sq = dx * dx + dy * dy;
                if (dist_sq > cell_radius * cell_radius) continue;

                int cx = imx + dx;
                int cy = imy + dy;

                if (cx < min_i || cx >= max_i || cy < min_j || cy >= max_j) continue;
                if (cx < 0 || cy < 0 ||
                    cx >= static_cast<int>(master_grid.getSizeInCellsX()) ||
                    cy >= static_cast<int>(master_grid.getSizeInCellsY())) continue;

                // 统一 cost: VLM 扩展区也用 254，让 inflation_layer 从新边缘膨胀
                unsigned char cell_cost = cost;

                unsigned char old_cost = master_grid.getCost(cx, cy);
                if (cell_cost > old_cost) {
                    master_grid.setCost(cx, cy, cell_cost);
                }
                marked_cells_.push_back(
                    static_cast<unsigned int>(cy) * master_grid.getSizeInCellsX() +
                    static_cast<unsigned int>(cx));
            }
        }
    }

    // ④ CMPT 权限交接：override 还活着但 obstacles_ 已过期的，用 override 坐标画圆
    {
        std::unordered_set<std::string> drawn_classes;
        for (const auto & obs : obstacles_) {
            drawn_classes.insert(obs.class_name);
        }

        std::lock_guard<std::mutex> olock(override_mutex_);
        for (const auto & [cls, ov] : radius_overrides_) {
            if (drawn_classes.count(cls) > 0) continue;
            if (std::abs(ov.map_x) < 1e-6 && std::abs(ov.map_y) < 1e-6) continue;

            double lambda = std::exp(-ov.neg_log_lambda);
            if (lambda <= 0.01) continue;

            auto cost_it = class_costs_.find(cls);
            if (cost_it == class_costs_.end()) continue;

            unsigned char cost = std::max(cost_it->second.cost, ov.cost);
            double r_def = cost_it->second.radius;
            double r_eff = std::max(r_def, r_def + lambda * (ov.radius - r_def));

            unsigned int mx, my;
            if (!master_grid.worldToMap(ov.map_x, ov.map_y, mx, my)) continue;

            int cell_radius = static_cast<int>(r_eff / master_grid.getResolution());
            if (cell_radius < 1) cell_radius = 1;
            int imx = static_cast<int>(mx);
            int imy = static_cast<int>(my);

            for (int dy = -cell_radius; dy <= cell_radius; dy++) {
                for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                    if (dx * dx + dy * dy > cell_radius * cell_radius) continue;
                    int cx = imx + dx;
                    int cy = imy + dy;
                    if (cx < min_i || cx >= max_i || cy < min_j || cy >= max_j) continue;
                    if (cx < 0 || cy < 0 ||
                        cx >= static_cast<int>(master_grid.getSizeInCellsX()) ||
                        cy >= static_cast<int>(master_grid.getSizeInCellsY())) continue;
                    unsigned char old_cost = master_grid.getCost(cx, cy);
                    if (cost > old_cost) {
                        master_grid.setCost(cx, cy, cost);
                    }
                    marked_cells_.push_back(
                        static_cast<unsigned int>(cy) * master_grid.getSizeInCellsX() +
                        static_cast<unsigned int>(cx));
                }
            }
        }
    }

    has_new_data_ = false;
}

// ═══════════════════════════════════════════════════
//  reset：清除所有缓存
// ═══════════════════════════════════════════════════
void SemanticDetectionLayer::reset()
{
    {
        std::lock_guard<std::mutex> lock(obstacles_mutex_);
        obstacles_.clear();
        marked_cells_.clear();
        has_new_data_ = false;
    }
    {
        std::lock_guard<std::mutex> olock(override_mutex_);
        radius_overrides_.clear();
    }
    {
        std::lock_guard<std::mutex> evidence_lock(evidence_mutex_);
        latest_class_evidence_.clear();
        latest_yolo_frame_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    }
}

}  // namespace wl100_semantic_costmap
