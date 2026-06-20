#ifndef WL100_CLEARANCE_PLANNER__CLEARANCE_A_STAR_PLANNER_HPP_
#define WL100_CLEARANCE_PLANNER__CLEARANCE_A_STAR_PLANNER_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav_msgs/msg/path.hpp"
#include "tf2_ros/buffer.h"

namespace wl100_clearance_planner
{

class ClearanceAStarPlanner : public nav2_core::GlobalPlanner
{
public:
  ClearanceAStarPlanner() = default;
  ~ClearanceAStarPlanner() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override;

private:
  struct SearchState
  {
    double g{0.0};
    double f{0.0};
    int parent{-1};
    bool closed{false};
    bool opened{false};
  };

  struct QueueEntry
  {
    unsigned int index{0};
    double f{0.0};
    double g{0.0};

    bool operator<(const QueueEntry & other) const
    {
      return f > other.f;
    }
  };

  void loadParameters();
  unsigned int toIndex(unsigned int mx, unsigned int my) const;
  std::pair<unsigned int, unsigned int> toCell(unsigned int index) const;
  bool isObstacleCost(unsigned char cost) const;
  bool isObstacleIndex(unsigned int index) const;
  bool isCellTraversable(
    unsigned int index,
    unsigned int start_index,
    double start_clearance,
    const std::vector<double> & clearance) const;
  bool worldToMapChecked(
    const geometry_msgs::msg::PoseStamped & pose,
    unsigned int & mx,
    unsigned int & my,
    const std::string & label) const;

  std::vector<double> computeClearanceField() const;
  bool findSafeGoal(
    unsigned int original_goal_index,
    const geometry_msgs::msg::PoseStamped & original_goal,
    const std::vector<double> & clearance,
    unsigned int & safe_goal_index,
    geometry_msgs::msg::PoseStamped & safe_goal) const;

  std::vector<unsigned int> runAStar(
    unsigned int start_index,
    unsigned int goal_index,
    const std::vector<double> & clearance) const;
  std::vector<unsigned int> reconstructPath(
    unsigned int start_index,
    unsigned int goal_index,
    const std::vector<SearchState> & states) const;

  double heuristic(unsigned int from_index, unsigned int to_index) const;
  double traversalCost(
    unsigned int from_index,
    unsigned int to_index,
    const std::vector<double> & clearance) const;

  bool lineIsSafe(
    unsigned int from_index,
    unsigned int to_index,
    const std::vector<double> & clearance) const;
  std::vector<unsigned int> shortcutPath(
    const std::vector<unsigned int> & raw_path,
    const std::vector<double> & clearance) const;

  nav_msgs::msg::Path buildPathMsg(
    const std::vector<unsigned int> & path_indices,
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal,
    const std::vector<double> & clearance) const;
  geometry_msgs::msg::Quaternion yawToQuaternion(double yaw) const;

  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_{nullptr};

  std::string name_;
  std::string global_frame_;
  unsigned int size_x_{0};
  unsigned int size_y_{0};
  double resolution_{0.05};

  double hard_clearance_m_{0.25};
  double preferred_clearance_m_{0.45};
  double clearance_weight_{12.0};
  double costmap_weight_{3.0};
  int lethal_cost_threshold_{253};
  bool unknown_is_obstacle_{true};
  double goal_search_radius_m_{0.50};
  int max_iterations_{1000000};
  bool shortcut_path_{true};
  double resample_step_m_{0.05};
};

}  // namespace wl100_clearance_planner

#endif  // WL100_CLEARANCE_PLANNER__CLEARANCE_A_STAR_PLANNER_HPP_
