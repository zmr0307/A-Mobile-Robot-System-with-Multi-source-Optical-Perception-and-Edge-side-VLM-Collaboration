#ifndef WL100_SEMANTIC_COSTMAP__SEMANTIC_DETECTION_LAYER_HPP_
#define WL100_SEMANTIC_COSTMAP__SEMANTIC_DETECTION_LAYER_HPP_

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/buffer.h"
#include "geometry_msgs/msg/point_stamped.hpp"

namespace wl100_semantic_costmap
{

/// 类别代价配置
struct ClassCostConfig
{
    unsigned char cost = 254;   // costmap 代价值 (0~254)
    double radius = 0.3;        // 物体占据半径 (m)
    double decay_time = 20.0;   // 该类别障碍物保持时间 (s)，动态物体设短
};

/// 缓存的语义障碍物（已投影到 odom 坐标系）
struct SemanticObstacle
{
    double x = 0.0;             // odom 坐标 X
    double y = 0.0;             // odom 坐标 Y
    std::string class_name;
    double confidence = 0.0;
    double distance = 0.0;      // 检测距离 (m)
    double bbox_width = 0.0;    // bbox 像素宽度
    rclcpp::Time stamp;         // 检测时间戳（用于衰减）
};

/// VLM 动态 radius/cost 覆盖（积分形式指数衰减，保证 P2 单调性 + CMPT 后验信任）
struct RadiusOverride
{
    double radius = 0.3;              // VLM 建议的膨胀半径
    unsigned char cost = 254;         // 根据 risk_level 映射的 cost
    double neg_log_lambda = 0.0;      // 累积衰减积分 ∫₀ᵗ ds/τ(s)，λ = exp(-neg_log_lambda)
    rclcpp::Time last_update_time;    // 上次积分步进时刻

    // CMPT 后验信任状态
    double psi_score = 0.0;               // 当前后验信任分 ψ
    rclcpp::Time last_evidence_stamp{0, 0, RCL_ROS_TIME};  // 上次处理到的 YOLO 证据时间
    double claimed_distance = -1.0;       // VLM override 创建时对应目标距离
    int miss_count = 0;                   // 连续 miss 计数，抑制 YOLO 短时漏检导致的误惩罚

    // P_vis（可见性先验）所需的障碍物 map 坐标
    double map_x = 0.0;                   // 障碍物在 global frame (odom) 中的最新 X 坐标
    double map_y = 0.0;                   // 障碍物在 global frame (odom) 中的最新 Y 坐标
};

/// 每一帧 YOLO 识别的最新类级证据摘要（用于 CMPT）
struct ClassEvidence
{
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    int count = 0;
    double best_conf = 0.0;
    double min_distance = 1e9;
    bool present = false;
};

/// Nav2 自定义 Costmap Layer：将 YOLOE-26 检测结果投影到代价地图
class SemanticDetectionLayer : public nav2_costmap_2d::Layer
{
public:
    SemanticDetectionLayer() = default;
    ~SemanticDetectionLayer() override = default;

    // ── Nav2 Layer 接口 ──
    void onInitialize() override;
    void updateBounds(
        double robot_x, double robot_y, double robot_yaw,
        double * min_x, double * min_y,
        double * max_x, double * max_y) override;
    void updateCosts(
        nav2_costmap_2d::Costmap2D & master_grid,
        int min_i, int min_j, int max_i, int max_j) override;
    void reset() override;
    bool isClearable() override { return true; }

private:
    // ── 回调 ──
    void detectionCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);

    // ── 工具 ──
    bool projectToOdom(
        double u, double v, double depth_m,
        const std::string & source_frame,
        const rclcpp::Time & stamp,
        double & out_x, double & out_y);
    void loadClassCosts();

    /// 计算可见性先验 P_vis：障碍物 map 坐标是否在当前传感器视野内
    /// 返回 [0, 1]，1 = 完全可见（在 FOV 正中），0 = 完全不可见（在视野外）
    double computeVisibilityPrior(double obs_map_x, double obs_map_y);

    // VLM override 回调
    void overrideCallback(const std_msgs::msg::String::SharedPtr msg);
    // risk_level → cost 映射
    static unsigned char riskToCost(double risk);

    // ── 参数 ──
    std::string detection_topic_;
    double decay_time_;           // YOLO 障碍物保持时间 (秒)
    double vlm_tau_base_ = 10.0;  // VLM 衰减基础时间常数 (s)
    double tau_beta_ = 0.5;       // 自适应 τ 调节系数: τ_eff = τ_base/(1+β·N)
    double tau_min_ = 3.0;        // CMPT 低信任下的最小衰减常数
    double tau_max_ = 20.0;       // CMPT 高信任下的最大衰减常数
    double psi_min_ = -6.0;       // ψ 下界，防数值发散
    double psi_max_ = 6.0;        // ψ 上界，防数值发散
    double gamma_ = 1.0;          // ψ -> trust 的 sigmoid 斜率
    double log_lr_pos_ = 0.8;     // 正证据奖励
    double log_lr_neg_ = 0.8;     // 负证据惩罚
    double yolo_stale_timeout_ = 1.0;       // YOLO 超时阈值
    double distance_consistency_scale_ = 0.8;  // 距离一致性衰减尺度
    double conf_min_ = 0.4;       // CMPT 置信度归一化下限
    std::atomic<int> n_dynamic_{0};  // YOLO 最近一帧动态物体数 (person+robot)
    double min_confidence_;       // 最低置信度过滤
    double transform_tolerance_;
    double max_detection_range_;  // 最大作用距离 (m)

    // D435i 相机内参 (640×480 实际标定值)
    double fx_ = 611.74;
    double fy_ = 611.69;
    double cx_ = 312.84;
    double cy_ = 243.38;

    // P_vis 传感器视场参数（由硬件规格确定，非算法超参数）
    double sensor_half_hfov_ = 0.759;  // D435i 水平半视角 43.5° = 0.759 rad
    double pvis_fade_width_ = 0.262;   // FOV 边缘过渡带宽度 15° = 0.262 rad

    // ── 类别代价映射 ──
    std::unordered_map<std::string, ClassCostConfig> class_costs_;

    // ── 检测缓存（线程安全）──
    std::mutex obstacles_mutex_;
    std::vector<SemanticObstacle> obstacles_;

    // ── ROS ──
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub_;
    bool has_new_data_ = false;

    // ── YOLO 最新类级证据缓存（线程安全）──
    std::mutex evidence_mutex_;
    std::unordered_map<std::string, ClassEvidence> latest_class_evidence_;
    rclcpp::Time latest_yolo_frame_stamp_{0, 0, RCL_ROS_TIME};

    // ── VLM 动态覆盖（线程安全）──
    std::mutex override_mutex_;    // 独立于 obstacles_mutex_
    std::unordered_map<std::string, RadiusOverride> radius_overrides_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr override_sub_;
    std::string override_topic_;

    // ── 记录上一轮标记的格子索引，用于清除"幽灵障碍" ──
    std::vector<unsigned int> marked_cells_;

    // ── DRSS 状态发布（用于 rosbag 录制 CMPT 时序数据）──
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr drss_state_pub_;
};

}  // namespace wl100_semantic_costmap

#endif  // WL100_SEMANTIC_COSTMAP__SEMANTIC_DETECTION_LAYER_HPP_
