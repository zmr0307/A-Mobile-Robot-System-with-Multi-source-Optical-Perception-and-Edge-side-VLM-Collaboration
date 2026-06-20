// hdl localizaton ROS2 코드 1 
#include <mutex>
#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <std_srvs/srv/empty.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <pcl/filters/voxel_grid.h>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/msg/scan_matching_status.hpp>
#include <hdl_global_localization/srv/set_global_map.hpp>
#include <hdl_global_localization/srv/query_global_localization.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <sstream>


using namespace std;

namespace hdl_localization {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kRadToDeg = 180.0 / kPi;

double normalize_angle(double angle) {
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

double yaw_from_matrix(const Eigen::Matrix4f& pose) {
  return std::atan2(pose(1, 0), pose(0, 0));
}

double yaw_from_quat(const geometry_msgs::msg::Quaternion& q) {
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

std::string pose_summary(const Eigen::Matrix4f& pose) {
  std::ostringstream oss;
  oss << "x=" << pose(0, 3)
      << " y=" << pose(1, 3)
      << " z=" << pose(2, 3)
      << " yaw=" << yaw_from_matrix(pose) * kRadToDeg << "deg";
  return oss.str();
}

Eigen::Matrix4f pose_msg_to_matrix(const geometry_msgs::msg::Pose& pose_msg) {
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  Eigen::Quaternionf q(
    pose_msg.orientation.w,
    pose_msg.orientation.x,
    pose_msg.orientation.y,
    pose_msg.orientation.z);
  q.normalize();
  pose.linear() = q.toRotationMatrix();
  pose.translation() = Eigen::Vector3f(
    pose_msg.position.x,
    pose_msg.position.y,
    pose_msg.position.z);
  return pose.matrix();
}
}

class HdlLocalizationNodelet : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet(const rclcpp::NodeOptions& options)
  : Node("hdl_localization", options)
  {
    tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    robot_odom_frame_id = declare_parameter<std::string>("robot_odom_frame_id", "robot_odom");
    odom_child_frame_id = declare_parameter<std::string>("odom_child_frame_id", "base_link");
    send_tf_transforms = declare_parameter<bool>("send_tf_transforms", true);
    cool_time_duration = declare_parameter<double>("cool_time_duration", 0.5);
    relocalize_num_candidates = declare_parameter<int>("relocalize_num_candidates", 80);
    relocalize_bbs_top_rank_limit = declare_parameter<int>("relocalize_bbs_top_rank_limit", 5);
    relocalize_bbs_score_ratio_thresh = declare_parameter<double>("relocalize_bbs_score_ratio_thresh", 0.85);
    relocalize_validation_score_thresh = declare_parameter<double>("relocalize_validation_score_thresh", 1.0);
    relocalize_validation_max_xy = declare_parameter<double>("relocalize_validation_max_xy", 0.35);
    relocalize_validation_max_yaw = declare_parameter<double>("relocalize_validation_max_yaw_deg", 8.0) / kRadToDeg;
    relocalize_validation_quality_thresh = declare_parameter<double>("relocalize_validation_quality_thresh", 1.2);
    relocalize_validation_xy_weight = declare_parameter<double>("relocalize_validation_xy_weight", 1.0);
    relocalize_validation_yaw_weight = declare_parameter<double>("relocalize_validation_yaw_weight", 0.02);
    relocalize_validation_inlier_dist = declare_parameter<double>("relocalize_validation_inlier_dist", 0.35);
    relocalize_validation_min_inlier_ratio = declare_parameter<double>("relocalize_validation_min_inlier_ratio", 0.55);
    relocalize_stability_window_size = std::max(1, static_cast<int>(declare_parameter<int>("relocalize_stability_window_size", 12)));
    relocalize_stability_required_passes = std::max(1, static_cast<int>(declare_parameter<int>("relocalize_stability_required_passes", 3)));
    relocalize_stability_vote_window_size = std::max(
      relocalize_stability_required_passes,
      static_cast<int>(declare_parameter<int>("relocalize_stability_vote_window_size", 8)));
    relocalize_stability_min_frames = std::max(relocalize_stability_window_size, static_cast<int>(declare_parameter<int>("relocalize_stability_min_frames", 80)));
    relocalize_stability_max_frames = std::max(
      relocalize_stability_min_frames + relocalize_stability_required_passes,
      static_cast<int>(declare_parameter<int>("relocalize_stability_max_frames", 360)));
    relocalize_stability_xy_std = declare_parameter<double>("relocalize_stability_xy_std", 0.035);
    relocalize_stability_yaw_std = declare_parameter<double>("relocalize_stability_yaw_std_deg", 0.9) / kRadToDeg;
    relocalize_stability_mean_shift_xy = declare_parameter<double>("relocalize_stability_mean_shift_xy", 0.02);
    relocalize_stability_mean_shift_yaw = declare_parameter<double>("relocalize_stability_mean_shift_yaw_deg", 0.5) / kRadToDeg;
    reg_method = declare_parameter<std::string>("reg_method", "NDT_OMP");
    ndt_neighbor_search_method = declare_parameter<std::string>("ndt_neighbor_search_method", "DIRECT7");
    ndt_neighbor_search_radius = declare_parameter<double>("ndt_neighbor_search_radius", 2.0);
    ndt_resolution = declare_parameter<double>("ndt_resolution", 1.0);
    enable_robot_odometry_prediction = declare_parameter<bool>("enable_robot_odometry_prediction", false);

    use_imu = declare_parameter<bool>("use_imu", true);
    invert_acc = declare_parameter<bool>("invert_acc", false);
    invert_gyro = declare_parameter<bool>("invert_gyro", false);
    if (use_imu) {
      RCLCPP_INFO(get_logger(), "enable imu-based prediction");
      imu_sub = create_subscription<sensor_msgs::msg::Imu>("/gpsimu_driver/imu_data", 256, std::bind(&HdlLocalizationNodelet::imu_callback, this, std::placeholders::_1));
    }
    points_sub = create_subscription<sensor_msgs::msg::PointCloud2>("/velodyne_points", 5, std::bind(&HdlLocalizationNodelet::points_callback, this, std::placeholders::_1));

    auto latch_qos = rclcpp::QoS(1).transient_local();
    globalmap_sub = create_subscription<sensor_msgs::msg::PointCloud2>("/globalmap", latch_qos, std::bind(&HdlLocalizationNodelet::globalmap_callback, this, std::placeholders::_1));

    initialpose_sub = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 8, std::bind(&HdlLocalizationNodelet::initialpose_callback, this, std::placeholders::_1));

    pose_pub = create_publisher<nav_msgs::msg::Odometry>("/odom", 5);
    aligned_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/aligned_points", 5);
    status_pub = create_publisher<msg::ScanMatchingStatus>("/status", 5);
    relocalize_succeeded_pub = create_publisher<std_msgs::msg::Bool>("/hdl_localization/relocalize_succeeded", 10);

    // global localization
    use_global_localization = declare_parameter<bool>("use_global_localization", true);
    if (use_global_localization) {
      RCLCPP_INFO_STREAM(get_logger(), "wait for global localization services");
      set_global_map_service = create_client<hdl_global_localization::srv::SetGlobalMap>("/hdl_global_localization/set_global_map");
      query_global_localization_service = create_client<hdl_global_localization::srv::QueryGlobalLocalization>("/hdl_global_localization/query");
      while (!set_global_map_service->wait_for_service(std::chrono::milliseconds(1000))) {
        RCLCPP_WARN(get_logger(), "Waiting for SetGlobalMap service");
        if (!rclcpp::ok()) {
          return;
        }
      }
      while (!query_global_localization_service->wait_for_service(std::chrono::milliseconds(1000))) {
        RCLCPP_WARN(get_logger(), "Waiting for QueryGlobalLocalization service");
        if (!rclcpp::ok()) {
          return;
        }
      }

      relocalize_server = create_service<std_srvs::srv::Empty>("/relocalize", std::bind(&HdlLocalizationNodelet::relocalize, this, std::placeholders::_1, std::placeholders::_2));
    }
    initialize_params();
  }

private:
  pcl::Registration<PointT, PointT>::Ptr create_registration() {
    if(reg_method == "NDT_OMP") {
      RCLCPP_INFO(get_logger(), "NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);
      ndt->setResolution(ndt_resolution);
      if (ndt_neighbor_search_method == "DIRECT1") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          RCLCPP_INFO(get_logger(), "search_method KDTREE is selected");
        } else {
          RCLCPP_WARN(get_logger(), "invalid search method was given");
          RCLCPP_WARN(get_logger(), "default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    } else if(reg_method.find("NDT_CUDA") != std::string::npos) {
      RCLCPP_INFO(get_logger(), "NDT_CUDA is selected");
      std::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if(reg_method.find("D2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
      } else if (reg_method.find("P2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
      }

      if (ndt_neighbor_search_method == "DIRECT1") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT1 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT7 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      } else if (ndt_neighbor_search_method == "DIRECT_RADIUS") {
        RCLCPP_INFO_STREAM(get_logger(), "search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
      } else {
        RCLCPP_WARN(get_logger(), "invalid search method was given");
      }
      return ndt;
    }

    RCLCPP_ERROR_STREAM(get_logger(), "unknown registration method:" << reg_method);
    return nullptr;
  }

  void initialize_params() {
    // intialize scan matching method
    double downsample_resolution = declare_parameter<double>("downsample_resolution", 0.1);
    std::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = voxelgrid;

    RCLCPP_INFO(get_logger(), "create registration method for localization");
    registration = create_registration();

    // global localization
    RCLCPP_INFO(get_logger(), "create registration method for fallback during relocalization");
    relocalizing = false;
    delta_estimater.reset(new DeltaEstimater(create_registration()));

    // initialize pose estimator
    bool specify_init_pose = declare_parameter<bool>("specify_init_pose", true);
    if (specify_init_pose) {
      RCLCPP_INFO(get_logger(), "initialize pose estimator with specified parameters!!");
      pose_estimator.reset(new hdl_localization::PoseEstimator(registration,
        get_clock()->now(),
        Eigen::Vector3f(declare_parameter<double>("init_pos_x", 0.0), declare_parameter<double>("init_pos_y", 0.0), declare_parameter<double>("init_pos_z", 0.0)),
        Eigen::Quaternionf(declare_parameter<double>("init_ori_w", 1.0), declare_parameter<double>("init_ori_x", 0.0), declare_parameter<double>("init_ori_y", 0.0), declare_parameter<double>("init_ori_z", 0.0)),
        cool_time_duration
      ));
    }
  }

private:

  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg) {
    // RCLCPP_INFO(get_logger(), "----------------"); 
    // RCLCPP_INFO(get_logger(), "imu_callback"); 
    // RCLCPP_INFO(get_logger(), "----------------"); 
    std::lock_guard<std::mutex> lock(imu_data_mutex);
    imu_data.push_back(imu_msg);
  }


  void points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr points_msg) {
    // RCLCPP_INFO(get_logger(), ""); 
    // RCLCPP_INFO(get_logger(), "points_callback"); 
    // RCLCPP_INFO(get_logger(), ""); 


    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    if (!pose_estimator) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5.0, "waiting for initial pose input!!");
      return;
    }

    if (!globalmap) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5.0, "globalmap has not been received!!");
      return;
    }

    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    // point_msg의 sensor_msg/pointCloud2 type을 pcl_cloud type으로 형변환
    // sensor_msg/pointCloud2 -> pcl::PointCloud<PointT>
    pcl::fromROSMsg(*points_msg, *pcl_cloud);

    if (pcl_cloud->empty()) {
      RCLCPP_ERROR(get_logger(), "cloud is empty!!");
      return;
    }

    // transform pointcloud into odom_child_frame_id
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, *tf_buffer)) {
        RCLCPP_ERROR(get_logger(), "point cloud cannot be transformed into target frame!!");
        return;
    }

    auto filtered = downsample(cloud);
    last_scan = filtered;

    if (relocalizing) {
      delta_estimater->add_frame(filtered);
    }

    Eigen::Matrix4f before = pose_estimator->matrix();

    // predict
    if (!use_imu) {
      pose_estimator->predict(stamp);
    } else {
      
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      // RCLCPP_INFO(get_logger(),"imu size is : %d ", imu_data.size());
      auto imu_iter = imu_data.begin();
      for (imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        if (rclcpp::Time(stamp) < rclcpp::Time((*imu_iter)->header.stamp)) {
          break;
        }
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        double acc_sign = invert_acc ? -1.0 : 1.0;
        double gyro_sign = invert_gyro ? -1.0 : 1.0;
        pose_estimator->predict((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      imu_data.erase(imu_data.begin(), imu_iter);
    }

    // odometry-based prediction
    rclcpp::Time last_correction_time = pose_estimator->last_correction_time();
    if (enable_robot_odometry_prediction && last_correction_time != rclcpp::Time((int64_t)0, get_clock()->get_clock_type())) {
      geometry_msgs::msg::TransformStamped odom_delta;
      if (tf_buffer->canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, rclcpp::Duration(std::chrono::milliseconds(100)))) {
        odom_delta = tf_buffer->lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, rclcpp::Duration(std::chrono::milliseconds(0)));
      } else if(tf_buffer->canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, rclcpp::Time((int64_t)0, get_clock()->get_clock_type()), robot_odom_frame_id, rclcpp::Duration(std::chrono::milliseconds(0)))) {
        odom_delta = tf_buffer->lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, rclcpp::Time((int64_t)0, get_clock()->get_clock_type()), robot_odom_frame_id, rclcpp::Duration(std::chrono::milliseconds(0)));
      }

      if(odom_delta.header.stamp == rclcpp::Time((int64_t)0, get_clock()->get_clock_type())) {
        RCLCPP_WARN_STREAM(get_logger(), "failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
      } else {
        Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
        pose_estimator->predict_odom(delta.cast<float>().matrix());
      }
    }

    // 문제가 되는 구문
    // correct
    auto aligned = pose_estimator->correct(stamp, filtered);

    if (pending_relocalize_stability_check) {
      if (!update_relocalize_stability_check()) {
        return;
      }
    }

    if(aligned_pub->get_subscription_count()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      sensor_msgs::msg::PointCloud2 aligned_msg;
      pcl::toROSMsg(*aligned, aligned_msg);
      aligned_pub->publish(aligned_msg);
    }

    if(status_pub->get_subscription_count()) {
      publish_scan_matching_status(points_msg->header, aligned);
    }

    publish_odometry(points_msg->header.stamp, pose_estimator->matrix());

    // RCLCPP_INFO(get_logger(), ""); 
    // RCLCPP_INFO(get_logger(), "----------finish points callback------------");
    // RCLCPP_INFO(get_logger(), "");
  }

  struct RelocalizeStabilityVote {
    bool candidate = false;
    double mean_x = 0.0;
    double mean_y = 0.0;
    double mean_yaw = 0.0;
  };

  bool update_relocalize_stability_check() {
    relocalize_stability_checked_frames++;

    const Eigen::Matrix4f pose = pose_estimator->matrix();
    const bool ndt_converged = registration->hasConverged();
    const double ndt_score = registration->getFitnessScore();

    relocalize_stability_window.push_back(pose);
    while (static_cast<int>(relocalize_stability_window.size()) > relocalize_stability_window_size) {
      relocalize_stability_window.pop_front();
    }

    if (static_cast<int>(relocalize_stability_window.size()) < relocalize_stability_window_size) {
      RCLCPP_INFO_STREAM(get_logger(),
        "[HDL重定位诊断] 重定位稳定确认填充 "
        << relocalize_stability_window.size() << "/" << relocalize_stability_window_size
        << " 当前(" << pose_summary(pose) << ")"
        << " 收敛=" << (ndt_converged ? "是" : "否")
        << " score=" << ndt_score);
      return false;
    }

    double mean_x = 0.0;
    double mean_y = 0.0;
    double mean_sin = 0.0;
    double mean_cos = 0.0;
    for (const auto& item : relocalize_stability_window) {
      mean_x += item(0, 3);
      mean_y += item(1, 3);
      const double yaw = yaw_from_matrix(item);
      mean_sin += std::sin(yaw);
      mean_cos += std::cos(yaw);
    }

    const double count = static_cast<double>(relocalize_stability_window.size());
    mean_x /= count;
    mean_y /= count;
    mean_sin /= count;
    mean_cos /= count;
    const double mean_yaw = std::atan2(mean_sin, mean_cos);

    double var_x = 0.0;
    double var_y = 0.0;
    double var_yaw = 0.0;
    for (const auto& item : relocalize_stability_window) {
      const double dx = item(0, 3) - mean_x;
      const double dy = item(1, 3) - mean_y;
      const double dyaw = normalize_angle(yaw_from_matrix(item) - mean_yaw);
      var_x += dx * dx;
      var_y += dy * dy;
      var_yaw += dyaw * dyaw;
    }

    const double std_x = std::sqrt(var_x / count);
    const double std_y = std::sqrt(var_y / count);
    const double std_yaw = std::sqrt(var_yaw / count);
    const bool window_stable =
      ndt_converged &&
      ndt_score <= relocalize_validation_score_thresh &&
      std_x <= relocalize_stability_xy_std &&
      std_y <= relocalize_stability_xy_std &&
      std_yaw <= relocalize_stability_yaw_std;
    const bool min_frames_ready = relocalize_stability_checked_frames >= relocalize_stability_min_frames;

    const bool vote_candidate = window_stable && min_frames_ready;
    if (min_frames_ready) {
      RelocalizeStabilityVote vote;
      vote.candidate = vote_candidate;
      vote.mean_x = mean_x;
      vote.mean_y = mean_y;
      vote.mean_yaw = mean_yaw;
      relocalize_stability_votes.push_back(vote);
      while (static_cast<int>(relocalize_stability_votes.size()) > relocalize_stability_vote_window_size) {
        relocalize_stability_votes.pop_front();
      }
    }

    int matching_votes = 0;
    double cluster_shift_xy = 0.0;
    double cluster_shift_yaw = 0.0;
    if (vote_candidate) {
      for (const auto& vote : relocalize_stability_votes) {
        if (!vote.candidate) {
          continue;
        }
        const double dx = mean_x - vote.mean_x;
        const double dy = mean_y - vote.mean_y;
        const double shift_xy = std::sqrt(dx * dx + dy * dy);
        const double shift_yaw = std::abs(normalize_angle(mean_yaw - vote.mean_yaw));
        if (shift_xy <= relocalize_stability_mean_shift_xy &&
            shift_yaw <= relocalize_stability_mean_shift_yaw) {
          matching_votes++;
          cluster_shift_xy = std::max(cluster_shift_xy, shift_xy);
          cluster_shift_yaw = std::max(cluster_shift_yaw, shift_yaw);
        }
      }
    }
    relocalize_stability_pass_count = matching_votes;
    const bool stable = vote_candidate && matching_votes >= relocalize_stability_required_passes;

    RCLCPP_INFO_STREAM(get_logger(),
      "[HDL重定位诊断] 重定位稳定确认："
      << "mean=(" << mean_x << ", " << mean_y << ")"
      << " yaw=" << mean_yaw * kRadToDeg << "deg"
      << " σx=" << std_x * 100.0 << "cm"
      << " σy=" << std_y * 100.0 << "cm"
      << " σyaw=" << std_yaw * kRadToDeg << "deg"
      << " score=" << ndt_score
      << " 收敛=" << (ndt_converged ? "是" : "否")
      << " 帧数=" << relocalize_stability_checked_frames << "/" << relocalize_stability_min_frames
      << " 候选稳定=" << (vote_candidate ? "是" : "否")
      << " 聚类漂移XY=" << cluster_shift_xy * 100.0 << "cm"
      << " 聚类漂移Yaw=" << cluster_shift_yaw * kRadToDeg << "deg"
      << " 投票通过=" << matching_votes << "/" << relocalize_stability_required_passes
      << " 投票窗口=" << relocalize_stability_votes.size() << "/" << relocalize_stability_vote_window_size
      << " 结果=" << (stable ? "通过" : "等待"));

    if (!window_stable || !min_frames_ready) {
      if (relocalize_stability_checked_frames >= relocalize_stability_max_frames) {
        return reject_relocalize_stability_timeout();
      }
      return false;
    }

    if (stable) {
      pending_relocalize_stability_check = false;
      relocalize_stability_window.clear();
      relocalize_stability_votes.clear();
      relocalize_stability_pass_count = 0;
      RCLCPP_INFO_STREAM(get_logger(),
        "[HDL重定位诊断] 稳定窗口投票确认通过，Global relocalization succeeded! "
        << "投票通过=" << matching_votes << "/" << relocalize_stability_required_passes
        << " 聚类漂移XY=" << cluster_shift_xy * 100.0 << "cm"
        << " 聚类漂移Yaw=" << cluster_shift_yaw * kRadToDeg << "deg");
      std_msgs::msg::Bool succeeded_msg;
      succeeded_msg.data = true;
      relocalize_succeeded_pub->publish(succeeded_msg);
      return true;
    }

    if (vote_candidate) {
      RCLCPP_INFO_STREAM(get_logger(),
        "[HDL重定位诊断] 稳定窗口候选通过 "
        << matching_votes << "/" << relocalize_stability_required_passes
        << "，继续等待投票确认");
      if (relocalize_stability_checked_frames >= relocalize_stability_max_frames) {
        return reject_relocalize_stability_timeout();
      }
      return false;
    }

    if (relocalize_stability_checked_frames >= relocalize_stability_max_frames) {
      return reject_relocalize_stability_timeout();
    }

    return false;
  }

  bool reject_relocalize_stability_timeout() {
    pending_relocalize_stability_check = false;
    relocalize_stability_window.clear();
    relocalize_stability_votes.clear();
    relocalize_stability_pass_count = 0;
    pose_estimator.reset();
    RCLCPP_ERROR_STREAM(get_logger(),
      "[HDL重定位诊断] 拒绝HDL重定位结果："
      << relocalize_stability_checked_frames
      << "帧内未通过稳定确认，已清空HDL位姿估计，等待重新触发/relocalize");
    return false;
  }

  /**
   * @brief callback for globalmap input
   * @param points_msg
   */
  void globalmap_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr points_msg) {
    RCLCPP_INFO(get_logger(), "");
    RCLCPP_INFO(get_logger(), "globalmap received!");
    RCLCPP_INFO(get_logger(), "");

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    // pcl::PointCloud<PointT>::Ptr cloud = boost::make_shared<pcl::PointCloud<PointT>>();
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;


    registration->setInputTarget(globalmap);

    if(use_global_localization) {
      RCLCPP_INFO(get_logger(), "set globalmap for global localization!");
      auto req  = std::make_shared<hdl_global_localization::srv::SetGlobalMap::Request>();
      pcl::toROSMsg(*globalmap, req->global_map);
      set_global_map_service->async_send_request(req,
        [this](rclcpp::Client<hdl_global_localization::srv::SetGlobalMap>::SharedFuture future) {
          try {
            future.get();
            RCLCPP_INFO(get_logger(), "Global map set for global localization successfully");
          } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to set global map: %s", e.what());
          }
        });
    }
  }

  struct RelocalizeCandidateValidation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    size_t index = 0;
    double bbs_score = 0.0;
    bool converged = false;
    bool accepted = false;
    double ndt_score = std::numeric_limits<double>::infinity();
    double quality_score = std::numeric_limits<double>::infinity();
    double selection_score = std::numeric_limits<double>::infinity();
    double delta_xy = std::numeric_limits<double>::infinity();
    double delta_yaw = std::numeric_limits<double>::infinity();
    double inlier_ratio = 0.0;
    std::string source;
    Eigen::Matrix4f initial_pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f refined_pose = Eigen::Matrix4f::Identity();
  };

  RelocalizeCandidateValidation validate_relocalize_candidate(
    size_t index,
    const Eigen::Matrix4f& initial_pose,
    double bbs_score,
    const std::string& source,
    double source_penalty,
    const pcl::PointCloud<PointT>::ConstPtr& scan,
    const pcl::PointCloud<PointT>::ConstPtr& map) {
    RelocalizeCandidateValidation validation;
    validation.index = index;
    validation.bbs_score = bbs_score;
    validation.source = source;
    validation.initial_pose = initial_pose;

    auto validator = create_registration();
    if (!validator) {
      return validation;
    }

    validator->setInputTarget(map);
    validator->setInputSource(scan);

    pcl::PointCloud<PointT> aligned;
    validator->align(aligned, validation.initial_pose);

    validation.refined_pose = validator->getFinalTransformation();
    if (!aligned.empty() && validator->getSearchMethodTarget()) {
      const double inlier_dist_sq = relocalize_validation_inlier_dist * relocalize_validation_inlier_dist;
      int inliers = 0;
      std::vector<int> k_indices;
      std::vector<float> k_sq_dists;
      for (const auto& point : aligned) {
        if (validator->getSearchMethodTarget()->nearestKSearch(point, 1, k_indices, k_sq_dists) > 0 &&
            !k_sq_dists.empty() &&
            k_sq_dists[0] <= inlier_dist_sq) {
          inliers++;
        }
      }
      validation.inlier_ratio = static_cast<double>(inliers) / static_cast<double>(aligned.size());
    }

    const Eigen::Matrix4f delta = validation.initial_pose.inverse() * validation.refined_pose;
    validation.delta_xy = delta.block<2, 1>(0, 3).norm();
    validation.delta_yaw = std::abs(normalize_angle(yaw_from_matrix(validation.refined_pose) - yaw_from_matrix(validation.initial_pose)));
    validation.converged = validator->hasConverged();
    validation.ndt_score = validator->getFitnessScore();
    validation.quality_score =
      validation.ndt_score +
      relocalize_validation_xy_weight * validation.delta_xy +
      relocalize_validation_yaw_weight * (validation.delta_yaw * kRadToDeg);
    validation.selection_score = validation.quality_score + source_penalty;
    validation.accepted =
      validation.converged &&
      std::isfinite(validation.ndt_score) &&
      std::isfinite(validation.quality_score) &&
      validation.ndt_score <= relocalize_validation_score_thresh &&
      validation.delta_xy <= relocalize_validation_max_xy &&
      validation.delta_yaw <= relocalize_validation_max_yaw &&
      validation.inlier_ratio >= relocalize_validation_min_inlier_ratio &&
      validation.quality_score <= relocalize_validation_quality_thresh;

    return validation;
  }

  RelocalizeCandidateValidation choose_better_relocalize_validation(
    const RelocalizeCandidateValidation& lhs,
    const RelocalizeCandidateValidation& rhs) const {
    if (lhs.accepted != rhs.accepted) {
      return lhs.accepted ? lhs : rhs;
    }
    if (lhs.selection_score != rhs.selection_score) {
      return lhs.selection_score < rhs.selection_score ? lhs : rhs;
    }
    return lhs.ndt_score <= rhs.ndt_score ? lhs : rhs;
  }

  /**
   * @brief perform global localization to relocalize the sensor position
   * @param
   */
  bool relocalize(std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res) {
    if(last_scan == nullptr) {
      RCLCPP_INFO_STREAM(get_logger(), "no scan has been received");
      return false;
    }

    relocalizing = true;
    delta_estimater->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    auto query_req  = std::make_shared<hdl_global_localization::srv::QueryGlobalLocalization::Request>();
    pcl::toROSMsg(*scan, query_req->cloud);
    query_req->max_num_candidates = std::max(1, relocalize_num_candidates);

    query_global_localization_service->async_send_request(query_req,
      [this, scan](rclcpp::Client<hdl_global_localization::srv::QueryGlobalLocalization>::SharedFuture future) {
        try {
          auto query_result = future.get();

          if (query_result->poses.empty()) {
            RCLCPP_ERROR(get_logger(), "QueryGlobalLocalization returned empty poses array");
            relocalizing = false;
            return;
          }

          RCLCPP_INFO_STREAM(get_logger(), "[HDL重定位诊断] BBS返回候选数: " << query_result->poses.size());
          for (size_t i = 0; i < query_result->poses.size(); i++) {
            const auto& candidate = query_result->poses[i];
            const double yaw = yaw_from_quat(candidate.orientation);
            const double error = i < query_result->errors.size() ? query_result->errors[i] : std::numeric_limits<double>::quiet_NaN();
            const double inlier = i < query_result->inlier_fractions.size() ? query_result->inlier_fractions[i] : std::numeric_limits<double>::quiet_NaN();
            RCLCPP_INFO_STREAM(get_logger(),
              "[HDL重定位诊断] BBS候选[" << i << "]"
              << " x=" << candidate.position.x
              << " y=" << candidate.position.y
              << " z=" << candidate.position.z
              << " yaw=" << yaw * kRadToDeg << "deg"
              << " error=" << error
              << " inlier=" << inlier);
          }

          pcl::PointCloud<PointT>::ConstPtr map = globalmap;
          if (!map) {
            RCLCPP_ERROR(get_logger(), "[HDL重定位诊断] 全局地图为空，无法验证BBS候选");
            relocalizing = false;
            return;
          }

          const Eigen::Isometry3f relocalize_delta = delta_estimater->estimated_delta();
          RelocalizeCandidateValidation best_candidate;
          bool has_best_candidate = false;
          double best_bbs_score = 0.0;
          for (const auto score : query_result->errors) {
            if (std::isfinite(score)) {
              best_bbs_score = std::max(best_bbs_score, score);
            }
          }
          if (best_bbs_score <= 0.0) {
            RCLCPP_ERROR(get_logger(), "[HDL重定位诊断] BBS最高分无效，拒绝本轮重定位");
            relocalizing = false;
            return;
          }

          RCLCPP_INFO_STREAM(get_logger(),
            "[HDL重定位诊断] 开始NDT复核BBS候选，score硬阈值=" << relocalize_validation_score_thresh
            << " BBS最高分=" << best_bbs_score
            << " BBS前N门槛=" << relocalize_bbs_top_rank_limit
            << " BBS相对分门槛=" << relocalize_bbs_score_ratio_thresh
            << " 修正XY硬阈值=" << relocalize_validation_max_xy * 100.0 << "cm"
            << " 修正Yaw硬阈值=" << relocalize_validation_max_yaw * kRadToDeg << "deg"
            << " inlier阈值=" << relocalize_validation_min_inlier_ratio
            << " inlier距离=" << relocalize_validation_inlier_dist * 100.0 << "cm"
            << " 综合质量阈值=" << relocalize_validation_quality_thresh);

          auto log_validation = [this](const RelocalizeCandidateValidation& validation) {
            RCLCPP_INFO_STREAM(get_logger(),
              "[HDL重定位诊断] 候选[" << validation.index << "] NDT复核(" << validation.source << ")："
              << "初值(" << pose_summary(validation.initial_pose) << ") "
              << "NDT后(" << pose_summary(validation.refined_pose) << ") "
              << "修正XY=" << validation.delta_xy * 100.0 << "cm"
              << " 修正Yaw=" << validation.delta_yaw * kRadToDeg << "deg"
              << " 收敛=" << (validation.converged ? "是" : "否")
              << " score=" << validation.ndt_score
              << " quality=" << validation.quality_score
              << " selection=" << validation.selection_score
              << " inlier=" << validation.inlier_ratio
              << " 结果=" << (validation.accepted ? "通过" : "拒绝"));
          };

          const Eigen::Matrix4f relocalize_delta_matrix = relocalize_delta.matrix();
          const double relocalize_delta_xy = relocalize_delta_matrix.block<2, 1>(0, 3).norm();
          const double relocalize_delta_yaw = std::abs(yaw_from_matrix(relocalize_delta_matrix));
          const bool relocalize_delta_large =
            relocalize_delta_xy > 0.5 ||
            relocalize_delta_yaw > 10.0 / kRadToDeg;
          const double relocalize_delta_penalty = relocalize_delta_large ? 0.5 : 0.05;
          RCLCPP_INFO_STREAM(get_logger(),
            "[HDL重定位诊断] relocalize_delta="
            << "XY=" << relocalize_delta_xy * 100.0 << "cm"
            << " Yaw=" << relocalize_delta_yaw * kRadToDeg << "deg"
            << " 状态=" << (relocalize_delta_large ? "过大，降权BBS+delta" : "正常"));

          for (size_t i = 0; i < query_result->poses.size(); i++) {
            const double bbs_score = i < query_result->errors.size() ? query_result->errors[i] : std::numeric_limits<double>::quiet_NaN();
            const double bbs_score_ratio = std::isfinite(bbs_score) && best_bbs_score > 0.0 ? bbs_score / best_bbs_score : 0.0;
            const bool bbs_gate_pass =
              static_cast<int>(i) < relocalize_bbs_top_rank_limit ||
              bbs_score_ratio >= relocalize_bbs_score_ratio_thresh;
            RCLCPP_INFO_STREAM(get_logger(),
              "[HDL重定位诊断] 候选[" << i << "] BBS门控："
              << "score=" << bbs_score
              << " ratio=" << bbs_score_ratio
              << " rank=" << i
              << " 结果=" << (bbs_gate_pass ? "通过" : "拒绝"));

            if (!bbs_gate_pass) {
              continue;
            }

            const Eigen::Matrix4f bbs_pose = pose_msg_to_matrix(query_result->poses[i]);
            const auto raw_validation = validate_relocalize_candidate(
              i,
              bbs_pose,
              bbs_score,
              "BBS原始",
              0.0,
              scan,
              map);
            const auto delta_validation = validate_relocalize_candidate(
              i,
              bbs_pose * relocalize_delta_matrix,
              bbs_score,
              "BBS+delta",
              relocalize_delta_penalty,
              scan,
              map);
            log_validation(raw_validation);
            log_validation(delta_validation);

            const auto validation = choose_better_relocalize_validation(raw_validation, delta_validation);

            RCLCPP_INFO_STREAM(get_logger(),
              "[HDL重定位诊断] 候选[" << validation.index << "] 选择NDT复核结果："
              << validation.source
              << " selection=" << validation.selection_score
              << " 位姿(" << pose_summary(validation.refined_pose) << ")"
              << " 结果=" << (validation.accepted ? "通过" : "拒绝"));

            if (!validation.accepted) {
              continue;
            }

            if (!has_best_candidate ||
                validation.selection_score < best_candidate.selection_score ||
                (validation.selection_score == best_candidate.selection_score && validation.ndt_score < best_candidate.ndt_score)) {
              best_candidate = validation;
              has_best_candidate = true;
            }
          }

          if (!has_best_candidate) {
            std::lock_guard<std::mutex> lock(pose_estimator_mutex);
            pending_relocalize_stability_check = false;
            relocalize_stability_checked_frames = 0;
            relocalize_stability_pass_count = 0;
            relocalize_stability_window.clear();
            relocalize_stability_votes.clear();
            pose_estimator.reset();
            RCLCPP_ERROR_STREAM(get_logger(),
              "[HDL重定位诊断] 拒绝HDL重定位结果："
              << query_result->poses.size()
              << " 个BBS候选全部未通过NDT复核，已清空HDL位姿估计，等待重新触发/relocalize");
            relocalizing = false;
            return;
          }

          const Eigen::Vector3f best_translation = best_candidate.refined_pose.block<3, 1>(0, 3);
          const Eigen::Matrix3f best_rotation = best_candidate.refined_pose.block<3, 3>(0, 0);
          {
            std::lock_guard<std::mutex> lock(pose_estimator_mutex);
            pose_estimator.reset(new hdl_localization::PoseEstimator(
              registration,
              get_clock()->now(),
              best_translation,
              Eigen::Quaternionf(best_rotation),
              cool_time_duration));

            relocalize_initial_pose = best_candidate.refined_pose;
            relocalize_stability_checked_frames = 0;
            relocalize_stability_pass_count = 0;
            relocalize_stability_window.clear();
            relocalize_stability_votes.clear();
            pending_relocalize_stability_check = true;
          }

          RCLCPP_INFO_STREAM(get_logger(), "--- Global localization result ---");
          RCLCPP_INFO_STREAM(get_logger(),
            "[HDL重定位诊断] 候选[" << best_candidate.index << "] 已进入多帧稳定确认："
            << "BBS score=" << best_candidate.bbs_score
            << " NDT score=" << best_candidate.ndt_score
            << " quality=" << best_candidate.quality_score
            << " inlier=" << best_candidate.inlier_ratio
            << " 位姿(" << pose_summary(best_candidate.refined_pose) << ")");
        } catch (const std::exception& e) {
          RCLCPP_ERROR(get_logger(), "Failed to query global localization: %s", e.what());
        }
        relocalizing = false;
      });

    return true;
  }

  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   * @param pose_msg
   */
  void initialpose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_msg) {
    RCLCPP_INFO(get_logger(), "initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    pose_estimator.reset(
          new hdl_localization::PoseEstimator(
            registration,
            get_clock()->now(),
            Eigen::Vector3f(p.x, p.y, p.z),
            Eigen::Quaternionf(q.w, q.x, q.y, q.z),
            cool_time_duration)
    );
  }


  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  void publish_odometry(const rclcpp::Time& stamp, const Eigen::Matrix4f& pose) {
    // broadcast the transform over tf
    if(send_tf_transforms) {
      if(tf_buffer->canTransform(robot_odom_frame_id, odom_child_frame_id, rclcpp::Time((int64_t)0, get_clock()->get_clock_type()))) {
        geometry_msgs::msg::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
        map_wrt_frame.header.stamp = stamp;
        map_wrt_frame.header.frame_id = odom_child_frame_id;
        map_wrt_frame.child_frame_id = "map";

        geometry_msgs::msg::TransformStamped frame_wrt_odom = tf_buffer->lookupTransform(robot_odom_frame_id, odom_child_frame_id, rclcpp::Time((int64_t)0, get_clock()->get_clock_type()), rclcpp::Duration(std::chrono::milliseconds(100)));
        Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

        geometry_msgs::msg::TransformStamped map_wrt_odom;
        tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

        tf2::Transform odom_wrt_map;
        tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
        odom_wrt_map = odom_wrt_map.inverse();

        geometry_msgs::msg::TransformStamped odom_trans;
        odom_trans.transform = tf2::toMsg(odom_wrt_map);
        odom_trans.header.stamp = stamp;
        odom_trans.header.frame_id = "map";
        odom_trans.child_frame_id = robot_odom_frame_id;

        tf_broadcaster->sendTransform(odom_trans);
      } else {
        geometry_msgs::msg::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
        odom_trans.header.stamp = stamp;
        odom_trans.header.frame_id = "map";
        odom_trans.child_frame_id = odom_child_frame_id;
        tf_broadcaster->sendTransform(odom_trans);
      }
    }

    // publish the transform
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";

    odom.pose.pose = tf2::toMsg(Eigen::Isometry3d(pose.cast<double>()));
    // odom.pose.pose.position.x = pose_trans.transform.translation.x;
    // odom.pose.pose.position.y = pose_trans.transform.translation.y;
    // odom.pose.pose.position.z = pose_trans.transform.translation.z;
    // odom.pose.pose.orientation = pose_trans.transform.rotation;
    odom.child_frame_id = odom_child_frame_id;
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    pose_pub->publish(odom);
  }

  void publish_scan_matching_status(const std_msgs::msg::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned) {
    msg::ScanMatchingStatus status;
    status.header = header;

    status.has_converged = registration->hasConverged();
    status.matching_error = registration->getFitnessScore();

    const double max_correspondence_dist = 0.5;

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for(int i = 0; i < aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();
    status.relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>())).transform;

    status.prediction_labels.reserve(2);
    status.prediction_errors.reserve(2);

    std::vector<double> errors(6, 0.0);

    if(pose_estimator->wo_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::msg::String());
      status.prediction_labels.back().data = "without_pred";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->wo_prediction_error().get().cast<double>())).transform);
    }

    if(pose_estimator->imu_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::msg::String());
      status.prediction_labels.back().data = use_imu ? "imu" : "motion_model";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_prediction_error().get().cast<double>())).transform);
    }

    if(pose_estimator->odom_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::msg::String());
      status.prediction_labels.back().data = "odom";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->odom_prediction_error().get().cast<double>())).transform);
    }

    status_pub->publish(status);
  }

private:
  std::string robot_odom_frame_id;
  std::string odom_child_frame_id;
  bool send_tf_transforms;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub;
  rclcpp::Publisher<hdl_localization::msg::ScanMatchingStatus>::SharedPtr status_pub;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr relocalize_succeeded_pub;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> imu_data;

  // globalmap and registration method
  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;

  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;
  Eigen::Matrix4f relocalize_initial_pose = Eigen::Matrix4f::Identity();
  bool pending_relocalize_stability_check = false;
  std::deque<Eigen::Matrix4f> relocalize_stability_window;
  std::deque<RelocalizeStabilityVote> relocalize_stability_votes;
  int relocalize_stability_checked_frames = 0;
  int relocalize_stability_pass_count = 0;

  // global localization
  bool use_global_localization;
  std::atomic_bool relocalizing;
  std::unique_ptr<DeltaEstimater> delta_estimater;

  pcl::PointCloud<PointT>::ConstPtr last_scan;
  rclcpp::Client<hdl_global_localization::srv::SetGlobalMap>::SharedPtr set_global_map_service;
  rclcpp::Client<hdl_global_localization::srv::QueryGlobalLocalization>::SharedPtr query_global_localization_service;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr relocalize_server;

  // Parameters
  double cool_time_duration;
  int relocalize_num_candidates;
  int relocalize_bbs_top_rank_limit;
  double relocalize_bbs_score_ratio_thresh;
  double relocalize_validation_score_thresh;
  double relocalize_validation_max_xy;
  double relocalize_validation_max_yaw;
  double relocalize_validation_quality_thresh;
  double relocalize_validation_xy_weight;
  double relocalize_validation_yaw_weight;
  double relocalize_validation_inlier_dist;
  double relocalize_validation_min_inlier_ratio;
  int relocalize_stability_window_size;
  int relocalize_stability_min_frames;
  int relocalize_stability_max_frames;
  int relocalize_stability_required_passes;
  int relocalize_stability_vote_window_size;
  double relocalize_stability_xy_std;
  double relocalize_stability_yaw_std;
  double relocalize_stability_mean_shift_xy;
  double relocalize_stability_mean_shift_yaw;
  std::string reg_method;
  std::string ndt_neighbor_search_method;
  double ndt_neighbor_search_radius;
  double ndt_resolution;
  bool enable_robot_odometry_prediction;
};
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hdl_localization::HdlLocalizationNodelet)
