#include <queue>
#include <deque>
#include <mutex>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include "localizers/commons.h"
#include "localizers/icp_localizer.h"
#include "interface/srv/relocalize.hpp"
#include "interface/srv/is_valid.hpp"
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr double kMaxYawJump = 0.3 / kRadToDeg;  // 0.3deg: reject yaw-only ICP jumps.

double normalizeAngle(double angle)
{
    while (angle > kPi)
        angle -= 2.0 * kPi;
    while (angle < -kPi)
        angle += 2.0 * kPi;
    return angle;
}

double yawFromRotation(const M3D &rotation)
{
    return std::atan2(rotation(1, 0), rotation(0, 0));
}

double yawDeltaAbs(const M3D &from, const M3D &to)
{
    return std::abs(normalizeAngle(yawFromRotation(to) - yawFromRotation(from)));
}

const char *yesNo(bool value)
{
    return value ? "是" : "否";
}

const char *rejectReasonToChinese(const std::string &reason)
{
    if (reason == "empty_map")
        return "地图未加载";
    if (reason == "rough_not_converged")
        return "粗匹配未收敛";
    if (reason == "rough_score_high")
        return "粗匹配分数过高";
    if (reason == "refine_not_converged")
        return "细匹配未收敛";
    if (reason == "refine_score_high")
        return "细匹配分数过高";
    if (reason == "accepted_by_icp")
        return "ICP已通过";
    if (reason == "not_run")
        return "尚未运行";
    return "未知原因";
}
}  // namespace

struct NodeConfig
{
    std::string cloud_topic = "/fastlio2/body_cloud";
    std::string odom_topic = "/fastlio2/lio_odom";
    std::string map_frame = "map";
    std::string local_frame = "lidar";
    double update_hz = 1.0;
    bool enable_continuous_icp_tf_update = true;
};

struct NodeState
{
    std::mutex message_mutex;
    std::mutex service_mutex;

    bool message_received = false;
    bool service_received = false;
    bool localize_success = false;
    bool settling_mode = false;
    int settling_accept_count = 0;
    bool has_tf_accept_time = false;
    int consecutive_tf_accepts = 0;
    int consecutive_tf_rejects = 0;
    double max_rejected_xy_since_accept = 0.0;
    double max_rejected_yaw_since_accept = 0.0;
    rclcpp::Time last_send_tf_time = rclcpp::Clock().now();
    rclcpp::Time last_tf_accept_time = rclcpp::Clock().now();
    builtin_interfaces::msg::Time last_message_time;
    CloudType::Ptr last_cloud = std::make_shared<CloudType>();
    M3D last_r;                          // localmap_body_r
    V3D last_t;                          // localmap_body_t
    M3D last_offset_r = M3D::Identity(); // map_localmap_r
    V3D last_offset_t = V3D::Zero();     // map_localmap_t
    std::deque<V3D> settling_offset_window;
    M4F initial_guess = M4F::Identity();
};

class LocalizerNode : public rclcpp::Node
{
public:
    LocalizerNode() : Node("localizer_node")
    {
        RCLCPP_INFO(this->get_logger(), "Localizer Node Started");
        loadParameters();
        rclcpp::QoS qos = rclcpp::QoS(10);
        m_cloud_sub.subscribe(this, m_config.cloud_topic, qos.get_rmw_qos_profile());
        m_odom_sub.subscribe(this, m_config.odom_topic, qos.get_rmw_qos_profile());

        m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

        m_sync = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>>>(message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>(10), m_cloud_sub, m_odom_sub);
        m_sync->setAgePenalty(0.1);
        m_sync->registerCallback(std::bind(&LocalizerNode::syncCB, this, std::placeholders::_1, std::placeholders::_2));
        m_localizer = std::make_shared<ICPLocalizer>(m_localizer_config);

        m_reloc_srv = this->create_service<interface::srv::Relocalize>("relocalize", std::bind(&LocalizerNode::relocCB, this, std::placeholders::_1, std::placeholders::_2));

        m_reloc_check_srv = this->create_service<interface::srv::IsValid>("relocalize_check", std::bind(&LocalizerNode::relocCheckCB, this, std::placeholders::_1, std::placeholders::_2));

        m_map_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("map_cloud", 10);

        m_timer = this->create_wall_timer(10ms, std::bind(&LocalizerNode::timerCB, this));
    }

    void loadParameters()
    {
        this->declare_parameter("config_path", "");
        std::string config_path;
        this->get_parameter<std::string>("config_path", config_path);
        YAML::Node config = YAML::LoadFile(config_path);
        if (!config)
        {
            RCLCPP_WARN(this->get_logger(), "FAIL TO LOAD YAML FILE!");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "LOAD FROM YAML CONFIG PATH: %s", config_path.c_str());

        m_config.cloud_topic = config["cloud_topic"].as<std::string>();
        m_config.odom_topic = config["odom_topic"].as<std::string>();
        m_config.map_frame = config["map_frame"].as<std::string>();
        m_config.local_frame = config["local_frame"].as<std::string>();
        m_config.update_hz = config["update_hz"].as<double>();
        if (config["enable_continuous_icp_tf_update"])
            m_config.enable_continuous_icp_tf_update = config["enable_continuous_icp_tf_update"].as<bool>();

        m_localizer_config.rough_scan_resolution = config["rough_scan_resolution"].as<double>();
        m_localizer_config.rough_map_resolution = config["rough_map_resolution"].as<double>();
        m_localizer_config.rough_max_iteration = config["rough_max_iteration"].as<int>();
        m_localizer_config.rough_score_thresh = config["rough_score_thresh"].as<double>();
        if (config["rough_max_correspondence_distance"])
            m_localizer_config.rough_max_correspondence_distance = config["rough_max_correspondence_distance"].as<double>();

        m_localizer_config.refine_scan_resolution = config["refine_scan_resolution"].as<double>();
        m_localizer_config.refine_map_resolution = config["refine_map_resolution"].as<double>();
        m_localizer_config.refine_max_iteration = config["refine_max_iteration"].as<int>();
        m_localizer_config.refine_score_thresh = config["refine_score_thresh"].as<double>();
        if (config["refine_max_correspondence_distance"])
            m_localizer_config.refine_max_correspondence_distance = config["refine_max_correspondence_distance"].as<double>();
        if (config["trimmed_overlap_ratio"])
            m_localizer_config.trimmed_overlap_ratio = config["trimmed_overlap_ratio"].as<double>();

        RCLCPP_INFO(this->get_logger(),
            "ICP参数：粗匹配对应距离=%.2fm，细匹配对应距离=%.2fm，Trimmed保留比例=%.2f",
            m_localizer_config.rough_max_correspondence_distance,
            m_localizer_config.refine_max_correspondence_distance,
            m_localizer_config.trimmed_overlap_ratio);
        RCLCPP_INFO(this->get_logger(),
            "持续ICP更新map->odom：%s",
            yesNo(m_config.enable_continuous_icp_tf_update));
    }
    void timerCB()
    {
        if (!m_state.message_received)
            return;

        rclcpp::Duration diff = rclcpp::Clock().now() - m_state.last_send_tf_time;

        bool update_tf = diff.seconds() > (1.0 / m_config.update_hz) && m_state.message_received;

        if (!update_tf)
        {
            sendBroadCastTF(m_state.last_message_time);
            return;
        }

        m_state.last_send_tf_time = rclcpp::Clock().now();

        if (!m_config.enable_continuous_icp_tf_update)
        {
            bool applied_initial_guess = false;
            builtin_interfaces::msg::Time current_time;
            {
                std::lock_guard<std::mutex> lock(m_state.message_mutex);
                current_time = m_state.last_message_time;
                if (m_state.service_received)
                {
                    M3D map_body_r = m_state.initial_guess.block<3, 3>(0, 0).cast<double>();
                    V3D map_body_t = m_state.initial_guess.block<3, 1>(0, 3).cast<double>();
                    m_state.last_offset_r = map_body_r * m_state.last_r.transpose();
                    m_state.last_offset_t =
                        -map_body_r * m_state.last_r.transpose() * m_state.last_t + map_body_t;
                    m_state.service_received = false;
                    m_state.localize_success = true;
                    m_state.consecutive_tf_accepts = 0;
                    m_state.consecutive_tf_rejects = 0;
                    m_state.max_rejected_xy_since_accept = 0.0;
                    m_state.max_rejected_yaw_since_accept = 0.0;
                    m_state.last_tf_accept_time = this->get_clock()->now();
                    m_state.has_tf_accept_time = true;
                    applied_initial_guess = true;
                }
            }

            if (applied_initial_guess)
            {
                RCLCPP_INFO(this->get_logger(),
                    "[Localizer] 已写入外部重定位初值，持续ICP TF更新已关闭，后续不再用ICP修改map->odom");
            }
            sendBroadCastTF(current_time);
            publishMapCloud(current_time);
            return;
        }

        M4F initial_guess = M4F::Identity();
        if (m_state.service_received)
        {
            std::lock_guard<std::mutex>(m_state.service_mutex);
            initial_guess = m_state.initial_guess;
            // m_state.service_received = false;
        }
        else
        {
            std::lock_guard<std::mutex>(m_state.message_mutex);
            initial_guess.block<3, 3>(0, 0) = (m_state.last_offset_r * m_state.last_r).cast<float>();
            initial_guess.block<3, 1>(0, 3) = (m_state.last_offset_r * m_state.last_t + m_state.last_offset_t).cast<float>();
        }

        M3D current_local_r;
        V3D current_local_t;
        builtin_interfaces::msg::Time current_time;
        {
            std::lock_guard<std::mutex>(m_state.message_mutex);
            current_local_r = m_state.last_r;
            current_local_t = m_state.last_t;
            current_time = m_state.last_message_time;
            m_localizer->setInput(m_state.last_cloud);
        }

        bool result = m_localizer->align(initial_guess);
        const ICPDebugInfo &debug_info = m_localizer->debugInfo();
        if (result)
        {
            M3D map_body_r = initial_guess.block<3, 3>(0, 0).cast<double>();
            V3D map_body_t = initial_guess.block<3, 1>(0, 3).cast<double>();
            M3D new_offset_r = map_body_r * current_local_r.transpose();
            V3D new_offset_t = -map_body_r * current_local_r.transpose() * current_local_t + map_body_t;

            auto calc_offset = [&](const M4F &map_body, M3D &offset_r, V3D &offset_t) {
                M3D stage_map_body_r = map_body.block<3, 3>(0, 0).cast<double>();
                V3D stage_map_body_t = map_body.block<3, 1>(0, 3).cast<double>();
                offset_r = stage_map_body_r * current_local_r.transpose();
                offset_t = -stage_map_body_r * current_local_r.transpose() * current_local_t + stage_map_body_t;
            };

            M3D rough_offset_r;
            V3D rough_offset_t;
            M3D refine_offset_r;
            V3D refine_offset_t;
            calc_offset(debug_info.rough_transform, rough_offset_r, rough_offset_t);
            calc_offset(debug_info.refine_transform, refine_offset_r, refine_offset_t);
            double rough_jump_xy = (rough_offset_t - m_state.last_offset_t).head<2>().norm();
            double rough_jump_yaw = yawDeltaAbs(m_state.last_offset_r, rough_offset_r);
            double refine_jump_xy = (refine_offset_t - m_state.last_offset_t).head<2>().norm();
            double refine_jump_yaw = yawDeltaAbs(m_state.last_offset_r, refine_offset_r);
            double rough_refine_xy = (refine_offset_t - rough_offset_t).head<2>().norm();
            double rough_refine_yaw = yawDeltaAbs(rough_offset_r, refine_offset_r);

            // ── 单阈值跳变保护 ──
            // 用户要求：ICP 帧间偏移只要超过 1.5cm 就直接拒绝。
            // 桥接初值在 relocalize service 中先写入 TF，后续这里统一按 1.5cm 限制。
            constexpr double kMaxOffsetJump = 0.015;          // 1.5cm 阈值
            V3D offset_delta = new_offset_t - m_state.last_offset_t;
            Eigen::Vector2d delta_xy = offset_delta.head<2>();
            double jump_dist = delta_xy.norm();  // 只看 XY 平面
            double yaw_delta = yawDeltaAbs(m_state.last_offset_r, new_offset_r);

            auto accept_update = [&]() {
                m_state.last_offset_r = new_offset_r;
                m_state.last_offset_t = new_offset_t;
                m_state.consecutive_tf_accepts++;
                m_state.consecutive_tf_rejects = 0;
                m_state.max_rejected_xy_since_accept = 0.0;
                m_state.max_rejected_yaw_since_accept = 0.0;
                m_state.last_tf_accept_time = this->get_clock()->now();
                m_state.has_tf_accept_time = true;
            };

            if (m_state.localize_success)
            {
                if (jump_dist > kMaxOffsetJump || yaw_delta > kMaxYawJump)
                {
                    m_state.consecutive_tf_rejects++;
                    m_state.consecutive_tf_accepts = 0;
                    m_state.max_rejected_xy_since_accept =
                        std::max(m_state.max_rejected_xy_since_accept, jump_dist);
                    m_state.max_rejected_yaw_since_accept =
                        std::max(m_state.max_rejected_yaw_since_accept, yaw_delta);
                    double seconds_since_accept = m_state.has_tf_accept_time ?
                        (this->get_clock()->now() - m_state.last_tf_accept_time).seconds() : -1.0;
                    const char *health_hint =
                        (m_state.consecutive_tf_rejects >= 10 ||
                         m_state.max_rejected_xy_since_accept > 0.10 ||
                         m_state.max_rejected_yaw_since_accept > (1.0 / kRadToDeg)) ?
                            "定位疑似失锁或当前区域匹配不稳定" :
                            "保护生效，暂不更新map->odom";

                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                        "[Localizer] ICP诊断：结果=拒绝TF跳变\n"
                        "  粗匹配：分数=%.5f / 阈值=%.5f，收敛=%s，点数=实时%zu / 地图%zu\n"
                        "  细匹配：分数=%.5f / 阈值=%.5f，收敛=%s，点数=实时%zu / 地图%zu\n"
                        "  粗匹配修正：XY=%.2fcm，Yaw=%.3fdeg\n"
                        "  细匹配修正：XY=%.2fcm，Yaw=%.3fdeg\n"
                        "  粗细差异：XY=%.2fcm，Yaw=%.3fdeg\n"
                        "  本次修正：XY=%.2fcm，Yaw=%.3fdeg\n"
                        "  拒绝阈值：XY=%.2fcm，Yaw=%.3fdeg\n"
                        "  健康状态：连续拒绝=%d次，上次接受后=%.1fs，最大被拒绝=XY %.2fcm / Yaw %.3fdeg，提示=%s",
                        debug_info.rough_score, m_localizer_config.rough_score_thresh,
                        yesNo(debug_info.rough_converged), debug_info.rough_input_size, debug_info.rough_target_size,
                        debug_info.refine_score, m_localizer_config.refine_score_thresh,
                        yesNo(debug_info.refine_converged), debug_info.refine_input_size, debug_info.refine_target_size,
                        rough_jump_xy * 100.0, rough_jump_yaw * kRadToDeg,
                        refine_jump_xy * 100.0, refine_jump_yaw * kRadToDeg,
                        rough_refine_xy * 100.0, rough_refine_yaw * kRadToDeg,
                        jump_dist * 100.0, yaw_delta * kRadToDeg,
                        kMaxOffsetJump * 100.0, kMaxYawJump * kRadToDeg,
                        m_state.consecutive_tf_rejects, seconds_since_accept,
                        m_state.max_rejected_xy_since_accept * 100.0,
                        m_state.max_rejected_yaw_since_accept * kRadToDeg,
                        health_hint);
                }
                else
                {
                    accept_update();
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                        "[Localizer] ICP诊断：结果=接受TF修正\n"
                        "  粗匹配：分数=%.5f / 阈值=%.5f，收敛=%s，点数=实时%zu / 地图%zu\n"
                        "  细匹配：分数=%.5f / 阈值=%.5f，收敛=%s，点数=实时%zu / 地图%zu\n"
                        "  粗匹配修正：XY=%.2fcm，Yaw=%.3fdeg\n"
                        "  细匹配修正：XY=%.2fcm，Yaw=%.3fdeg\n"
                        "  粗细差异：XY=%.2fcm，Yaw=%.3fdeg\n"
                        "  本次修正：XY=%.2fcm，Yaw=%.3fdeg\n"
                        "  接受条件：XY<=%.2fcm，Yaw<=%.3fdeg\n"
                        "  健康状态：连续接受=%d次，连续拒绝已清零",
                        debug_info.rough_score, m_localizer_config.rough_score_thresh,
                        yesNo(debug_info.rough_converged), debug_info.rough_input_size, debug_info.rough_target_size,
                        debug_info.refine_score, m_localizer_config.refine_score_thresh,
                        yesNo(debug_info.refine_converged), debug_info.refine_input_size, debug_info.refine_target_size,
                        rough_jump_xy * 100.0, rough_jump_yaw * kRadToDeg,
                        refine_jump_xy * 100.0, refine_jump_yaw * kRadToDeg,
                        rough_refine_xy * 100.0, rough_refine_yaw * kRadToDeg,
                        jump_dist * 100.0, yaw_delta * kRadToDeg,
                        kMaxOffsetJump * 100.0, kMaxYawJump * kRadToDeg,
                        m_state.consecutive_tf_accepts);
                }
            }
            else
            {
                accept_update();
                if (m_state.service_received)
                {
                    std::lock_guard<std::mutex>(m_state.service_mutex);
                    m_state.localize_success = true;
                    m_state.service_received = false;
                    RCLCPP_INFO(this->get_logger(),
                        "[Localizer] 桥接初始对齐已写入 TF，后续启用 1.5cm 跳变保护");
                }
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "[Localizer] ICP诊断：结果=初始TF写入\n"
                    "  粗匹配：分数=%.5f / 阈值=%.5f，收敛=%s，点数=实时%zu / 地图%zu\n"
                    "  细匹配：分数=%.5f / 阈值=%.5f，收敛=%s，点数=实时%zu / 地图%zu\n"
                    "  粗匹配修正：XY=%.2fcm，Yaw=%.3fdeg\n"
                    "  细匹配修正：XY=%.2fcm，Yaw=%.3fdeg\n"
                    "  粗细差异：XY=%.2fcm，Yaw=%.3fdeg\n"
                    "  初始修正：XY=%.2fcm，Yaw=%.3fdeg\n"
                    "  后续保护：XY>1.50cm 或 Yaw>0.300deg 时拒绝更新map->odom",
                    debug_info.rough_score, m_localizer_config.rough_score_thresh,
                    yesNo(debug_info.rough_converged), debug_info.rough_input_size, debug_info.rough_target_size,
                    debug_info.refine_score, m_localizer_config.refine_score_thresh,
                    yesNo(debug_info.refine_converged), debug_info.refine_input_size, debug_info.refine_target_size,
                    rough_jump_xy * 100.0, rough_jump_yaw * kRadToDeg,
                    refine_jump_xy * 100.0, refine_jump_yaw * kRadToDeg,
                    rough_refine_xy * 100.0, rough_refine_yaw * kRadToDeg,
                    jump_dist * 100.0, yaw_delta * kRadToDeg);
            }
        }
        else
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[Localizer] ICP诊断：结果=ICP失败\n"
                "  原因：%s\n"
                "  粗匹配：分数=%.5f / 阈值=%.5f，收敛=%s，点数=实时%zu / 地图%zu\n"
                "  细匹配：分数=%.5f / 阈值=%.5f，收敛=%s，点数=实时%zu / 地图%zu",
                rejectReasonToChinese(debug_info.reject_reason),
                debug_info.rough_score, m_localizer_config.rough_score_thresh,
                yesNo(debug_info.rough_converged), debug_info.rough_input_size, debug_info.rough_target_size,
                debug_info.refine_score, m_localizer_config.refine_score_thresh,
                yesNo(debug_info.refine_converged), debug_info.refine_input_size, debug_info.refine_target_size);
        }
        sendBroadCastTF(current_time);
        publishMapCloud(current_time);
    }
    void syncCB(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg, const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg)
    {

        std::lock_guard<std::mutex>(m_state.message_mutex);

        pcl::fromROSMsg(*cloud_msg, *m_state.last_cloud);

        m_state.last_r = Eigen::Quaterniond(odom_msg->pose.pose.orientation.w,
                                            odom_msg->pose.pose.orientation.x,
                                            odom_msg->pose.pose.orientation.y,
                                            odom_msg->pose.pose.orientation.z)
                             .toRotationMatrix();
        m_state.last_t = V3D(odom_msg->pose.pose.position.x,
                             odom_msg->pose.pose.position.y,
                             odom_msg->pose.pose.position.z);
        m_state.last_message_time = cloud_msg->header.stamp;
        if (!m_state.message_received)
        {
            m_state.message_received = true;
            m_config.local_frame = odom_msg->header.frame_id;
        }
    }

    void sendBroadCastTF(builtin_interfaces::msg::Time &time)
    {
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.frame_id = m_config.map_frame;
        transformStamped.child_frame_id = m_config.local_frame;
        transformStamped.header.stamp = time;
        Eigen::Quaterniond q(m_state.last_offset_r);
        V3D t = m_state.last_offset_t;
        transformStamped.transform.translation.x = t.x();
        transformStamped.transform.translation.y = t.y();
        transformStamped.transform.translation.z = t.z();
        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();
        m_tf_broadcaster->sendTransform(transformStamped);
    }

    void relocCB(const std::shared_ptr<interface::srv::Relocalize::Request> request, std::shared_ptr<interface::srv::Relocalize::Response> response)
    {
        std::string pcd_path = request->pcd_path;
        float x = request->x;
        float y = request->y;
        float z = request->z;
        float yaw = request->yaw;
        float roll = request->roll;
        float pitch = request->pitch;

        if (!std::filesystem::exists(pcd_path))
        {
            response->success = false;
            response->message = "pcd file not found";
            return;
        }

        Eigen::AngleAxisd yaw_angle = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd roll_angle = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch_angle = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY());
        bool load_flag = m_localizer->loadMap(pcd_path);
        if (!load_flag)
        {
            response->success = false;
            response->message = "load map failed";
            return;
        }
        {
            std::lock_guard<std::mutex>(m_state.message_mutex);
            m_state.initial_guess.setIdentity();
            m_state.initial_guess.block<3, 3>(0, 0) = (yaw_angle * roll_angle * pitch_angle).toRotationMatrix().cast<float>();
            m_state.initial_guess.block<3, 1>(0, 3) = V3F(x, y, z);

            // 先把桥接初值直接写入当前 offset，保证 RViz 立即跳到 HDL 给的位置。
            if (m_state.message_received)
            {
                M3D map_body_r = (yaw_angle * roll_angle * pitch_angle).toRotationMatrix();
                V3D map_body_t(x, y, z);
                m_state.last_offset_r = map_body_r * m_state.last_r.transpose();
                m_state.last_offset_t =
                    -map_body_r * m_state.last_r.transpose() * m_state.last_t + map_body_t;
            }

            m_state.service_received = !m_config.enable_continuous_icp_tf_update && !m_state.message_received;
            m_state.localize_success = !m_config.enable_continuous_icp_tf_update && m_state.message_received;
            if (m_config.enable_continuous_icp_tf_update)
            {
                m_state.service_received = true;
                m_state.localize_success = false;
            }
            if (!m_config.enable_continuous_icp_tf_update && m_state.message_received)
            {
                m_state.consecutive_tf_accepts = 0;
                m_state.consecutive_tf_rejects = 0;
                m_state.max_rejected_xy_since_accept = 0.0;
                m_state.max_rejected_yaw_since_accept = 0.0;
                m_state.last_tf_accept_time = this->get_clock()->now();
                m_state.has_tf_accept_time = true;
            }
        }

        if (!m_config.enable_continuous_icp_tf_update)
        {
            RCLCPP_INFO(this->get_logger(),
                "[Localizer] 外部重定位初值已写入TF，持续ICP TF更新关闭，不再用ICP修正map->odom");
        }

        response->success = true;
        response->message = "relocalize success";
        return;
    }

    void relocCheckCB(const std::shared_ptr<interface::srv::IsValid::Request> request, std::shared_ptr<interface::srv::IsValid::Response> response)
    {
        std::lock_guard<std::mutex>(m_state.service_mutex);
        if (request->code == 1)
            response->valid = true;
        else
            response->valid = m_state.localize_success;
        return;
    }
    void publishMapCloud(builtin_interfaces::msg::Time &time)
    {
        if (m_map_cloud_pub->get_subscription_count() < 1)
            return;
        CloudType::Ptr map_cloud = m_localizer->refineMap();
        if (map_cloud->size() < 1)
            return;
        sensor_msgs::msg::PointCloud2 map_cloud_msg;
        pcl::toROSMsg(*map_cloud, map_cloud_msg);
        map_cloud_msg.header.frame_id = m_config.map_frame;
        map_cloud_msg.header.stamp = time;
        m_map_cloud_pub->publish(map_cloud_msg);
    }

private:
    NodeConfig m_config;
    NodeState m_state;

    ICPConfig m_localizer_config;
    std::shared_ptr<ICPLocalizer> m_localizer;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> m_cloud_sub;
    message_filters::Subscriber<nav_msgs::msg::Odometry> m_odom_sub;
    rclcpp::TimerBase::SharedPtr m_timer;
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>>> m_sync;
    std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;
    rclcpp::Service<interface::srv::Relocalize>::SharedPtr m_reloc_srv;
    rclcpp::Service<interface::srv::IsValid>::SharedPtr m_reloc_check_srv;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_map_cloud_pub;
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalizerNode>());
    rclcpp::shutdown();
    return 0;
}
