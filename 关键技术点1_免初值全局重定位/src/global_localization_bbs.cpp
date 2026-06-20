#include <hdl_global_localization/engines/global_localization_bbs.hpp>

#include "pcl_conversions/pcl_conversions.h"
#include <sensor_msgs/msg/point_cloud.hpp>

#include <hdl_global_localization/bbs/bbs_localization.hpp>
#include <hdl_global_localization/bbs/occupancy_gridmap.hpp>

#include <algorithm>
#include <cmath>

namespace hdl_global_localization {

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

double yaw_from_pose(const Eigen::Isometry3f& pose) {
  return std::atan2(pose.linear()(1, 0), pose.linear()(0, 0));
}

bool is_unique_candidate(
  const Eigen::Isometry3f& candidate,
  const std::vector<GlobalLocalizationResult::Ptr>& selected,
  double min_xy,
  double min_yaw) {
  const double candidate_yaw = yaw_from_pose(candidate);

  for (const auto& result : selected) {
    const double diff_xy = (candidate.translation().head<2>() - result->pose.translation().head<2>()).norm();
    const double diff_yaw = std::abs(normalize_angle(candidate_yaw - yaw_from_pose(result->pose)));
    if (diff_xy < min_xy && diff_yaw < min_yaw) {
      return false;
    }
  }

  return true;
}
}

GlobalLocalizationBBS::GlobalLocalizationBBS(rclcpp::Node::SharedPtr node) : node(node) {
  gridmap_pub = this->node->create_publisher<nav_msgs::msg::OccupancyGrid>("gridmap", 1);
  map_slice_pub = this->node->create_publisher<sensor_msgs::msg::PointCloud2>("map_slice", 1);
  scan_slice_pub = this->node->create_publisher<sensor_msgs::msg::PointCloud2>("scan_slice", 1);

  params.max_range = this->node->declare_parameter<double>("bbs.max_range", 15.0);
  params.min_tx = this->node->declare_parameter<double>("bbs.min_tx", -50.0);
  params.max_tx = this->node->declare_parameter<double>("bbs.max_tx", 50.0);
  params.min_ty = this->node->declare_parameter<double>("bbs.min_ty", -50.0);
  params.max_ty = this->node->declare_parameter<double>("bbs.max_ty", 50.0);
  params.min_theta = this->node->declare_parameter<double>("bbs.min_theta", -3.15);
  params.max_theta = this->node->declare_parameter<double>("bbs.max_theta", 3.15);
  params.map_min_z = this->node->declare_parameter<double>("bbs.map_min_z", 0.2);
  params.map_max_z = this->node->declare_parameter<double>("bbs.map_max_z", 1.2);
  params.map_resolution = this->node->declare_parameter<double>("bbs.map_resolution", 0.5);
  params.scan_min_z = this->node->declare_parameter<double>("bbs.scan_min_z", 0.2);
  params.scan_max_z = this->node->declare_parameter<double>("bbs.scan_max_z", 1.2);
  params.map_width = this->node->declare_parameter<int>("bbs.map_width", 1024);
  params.map_height = this->node->declare_parameter<int>("bbs.map_height", 1024);
  params.map_pyramid_level = this->node->declare_parameter<int>("bbs.map_pyramid_level", 6);
  params.max_points_per_cell = this->node->declare_parameter<int>("bbs.max_points_per_cell", 5);
  params.candidate_distribution_enabled = this->node->declare_parameter<bool>("bbs.candidate_distribution_enabled", true);
  params.candidate_raw_multiplier = this->node->declare_parameter<int>("bbs.candidate_raw_multiplier", 20);
  params.candidate_max_leaf_evaluations = this->node->declare_parameter<int>("bbs.candidate_max_leaf_evaluations", 8000);
  params.candidate_min_xy = this->node->declare_parameter<double>("bbs.candidate_min_xy", 0.75);
  params.candidate_min_yaw = this->node->declare_parameter<double>("bbs.candidate_min_yaw", 8.0 / kRadToDeg);
  params.candidate_bucket_xy = this->node->declare_parameter<double>("bbs.candidate_bucket_xy", 2.0);
  params.candidate_bucket_yaw = this->node->declare_parameter<double>("bbs.candidate_bucket_yaw", 20.0 / kRadToDeg);
  params.candidate_min_score_ratio = this->node->declare_parameter<double>("bbs.candidate_min_score_ratio", 0.25);
}

GlobalLocalizationBBS ::~GlobalLocalizationBBS() {}

void GlobalLocalizationBBS::set_global_map(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
  BBSParams p;
  p.max_range = params.max_range;
  p.min_tx = params.min_tx;
  p.max_tx = params.max_tx;
  p.min_ty = params.min_ty;
  p.max_ty = params.max_ty;
  p.min_theta = params.min_theta;
  p.max_theta = params.max_theta;
  bbs.reset(new BBSLocalization(p));

  auto map_2d = slice(*cloud, params.map_min_z, params.map_max_z);
  RCLCPP_INFO_STREAM(node->get_logger(), "Set Map " << map_2d.size() << " points");

  if (map_2d.size() < 128) {
    RCLCPP_WARN_STREAM(node->get_logger(), "Num points in the sliced map is too small!!");
    RCLCPP_WARN_STREAM(node->get_logger(), "Change the slice range parameters!!");
  }

  bbs->set_map(map_2d, params.map_resolution, params.map_width, params.map_height, params.map_pyramid_level, params.max_points_per_cell);

  auto map_3d = unslice(map_2d);
  map_3d->header.frame_id = "map";
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(*map_3d, msg);
  map_slice_pub->publish(msg);
  gridmap_pub->publish(*bbs->gridmap()->to_rosmsg());
}

GlobalLocalizationResults GlobalLocalizationBBS::query(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, int max_num_candidates) {
  auto scan_2d = slice(*cloud, params.scan_min_z, params.scan_max_z);

  std::vector<GlobalLocalizationResult::Ptr> results;

  RCLCPP_INFO_STREAM(node->get_logger(), "Query " << scan_2d.size() << " points");
  if (scan_2d.size() < 32) {
    RCLCPP_WARN_STREAM(node->get_logger(), "Num points in the sliced scan is too small!!");
    RCLCPP_WARN_STREAM(node->get_logger(), "Change the slice range parameters!!");
    return GlobalLocalizationResults(results);
  }

  const int requested_candidates = std::max(1, max_num_candidates);
  const int raw_multiplier = std::max(1, params.candidate_raw_multiplier);
  const int raw_candidate_count = requested_candidates * raw_multiplier;
  const double min_candidate_score = scan_2d.size() * std::max(0.0, params.candidate_min_score_ratio);
  BBSLocalization::Results raw_candidates;
  if (params.candidate_distribution_enabled) {
    raw_candidates = bbs->localize_n_distributed(
      scan_2d,
      min_candidate_score,
      raw_candidate_count,
      params.candidate_bucket_xy,
      params.candidate_bucket_yaw,
      params.candidate_max_leaf_evaluations);
  } else {
    raw_candidates = bbs->localize_n(scan_2d, min_candidate_score, raw_candidate_count);
  }
  if (raw_candidates.empty()) {
    return GlobalLocalizationResults(results);
  }

  if (scan_slice_pub->get_subscription_count()) {
    auto scan_3d = unslice(scan_2d);
    scan_3d->header = cloud->header;
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*scan_3d, msg);
    scan_slice_pub->publish(msg);
  }

  RCLCPP_INFO_STREAM(node->get_logger(),
    "[BBS诊断] 请求候选数=" << max_num_candidates
    << "，原始搜索候选数=" << raw_candidate_count
    << "，原始返回=" << raw_candidates.size()
    << "，分布候选=" << (params.candidate_distribution_enabled ? "启用" : "关闭")
    << "，最低候选分=" << min_candidate_score
    << "，分桶XY=" << params.candidate_bucket_xy << "m"
    << " Yaw=" << params.candidate_bucket_yaw * kRadToDeg << "deg"
    << "，去重阈值XY=" << params.candidate_min_xy << "m"
    << " Yaw=" << params.candidate_min_yaw * kRadToDeg << "deg");

  for (size_t i = 0; i < raw_candidates.size(); i++) {
    Eigen::Isometry3f trans_3d = Eigen::Isometry3f::Identity();
    trans_3d.linear().block<2, 2>(0, 0) = raw_candidates[i].pose.linear();
    trans_3d.translation().head<2>() = raw_candidates[i].pose.translation();

    const double yaw = yaw_from_pose(trans_3d);
    const bool unique = is_unique_candidate(trans_3d, results, params.candidate_min_xy, params.candidate_min_yaw);
    RCLCPP_INFO_STREAM(node->get_logger(),
      "[BBS诊断] 原始候选[" << i << "]"
      << " x=" << trans_3d.translation().x()
      << " y=" << trans_3d.translation().y()
      << " yaw=" << yaw * kRadToDeg << "deg"
      << " score=" << raw_candidates[i].score
      << " unique=" << (unique ? "是" : "否"));

    if (!unique) {
      continue;
    }

    results.emplace_back(new GlobalLocalizationResult(raw_candidates[i].score, raw_candidates[i].score, trans_3d));
    if (static_cast<int>(results.size()) >= requested_candidates) {
      break;
    }
  }

  RCLCPP_INFO_STREAM(node->get_logger(),
    "[BBS诊断] 去重后返回候选数=" << results.size()
    << " / 请求=" << requested_candidates);

  return GlobalLocalizationResults(results);
}

GlobalLocalizationBBS::Points2D GlobalLocalizationBBS::slice(const pcl::PointCloud<pcl::PointXYZ>& cloud, double min_z, double max_z) const {
  Points2D points_2d;
  points_2d.reserve(cloud.size());
  for (int i = 0; i < cloud.size(); i++) {
    if (min_z < cloud.at(i).z && cloud.at(i).z < max_z) {
      points_2d.push_back(cloud.at(i).getVector3fMap().head<2>());
    }
  }
  return points_2d;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr GlobalLocalizationBBS::unslice(const Points2D& points) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->resize(points.size());
  for (int i = 0; i < points.size(); i++) {
    cloud->at(i).getVector3fMap().head<2>() = points[i];
    cloud->at(i).z = 0.0f;
  }

  return cloud;
}

}  // namespace hdl_global_localization
