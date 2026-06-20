#include "wl100_clearance_planner/clearance_a_star_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

#include "nav2_core/exceptions.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace wl100_clearance_planner
{
namespace
{
constexpr double kEpsilon = 1.0e-9;
constexpr double kSqrt2 = 1.4142135623730951;
constexpr int kNeighborCount = 8;
constexpr int kDx[kNeighborCount] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int kDy[kNeighborCount] = {0, 1, 1, 1, 0, -1, -1, -1};
}  // namespace

void ClearanceAStarPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock();
  if (!node_) {
    throw nav2_core::PlannerException("ClearanceAStarPlanner failed to lock lifecycle node");
  }

  name_ = std::move(name);
  tf_ = std::move(tf);
  costmap_ros_ = std::move(costmap_ros);
  costmap_ = costmap_ros_->getCostmap();
  global_frame_ = costmap_ros_->getGlobalFrameID();

  loadParameters();

  RCLCPP_INFO(
    node_->get_logger(),
    "Configured %s: hard_clearance=%.2fm preferred_clearance=%.2fm clearance_weight=%.2f "
    "costmap_weight=%.2f lethal_threshold=%d unknown_is_obstacle=%s",
    name_.c_str(), hard_clearance_m_, preferred_clearance_m_, clearance_weight_,
    costmap_weight_, lethal_cost_threshold_, unknown_is_obstacle_ ? "true" : "false");
}

void ClearanceAStarPlanner::cleanup()
{
  RCLCPP_INFO(node_->get_logger(), "Cleaning up %s", name_.c_str());
}

void ClearanceAStarPlanner::activate()
{
  RCLCPP_INFO(node_->get_logger(), "Activating %s", name_.c_str());
}

void ClearanceAStarPlanner::deactivate()
{
  RCLCPP_INFO(node_->get_logger(), "Deactivating %s", name_.c_str());
}

void ClearanceAStarPlanner::loadParameters()
{
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".hard_clearance_m", rclcpp::ParameterValue(0.25));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".preferred_clearance_m", rclcpp::ParameterValue(0.45));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".clearance_weight", rclcpp::ParameterValue(12.0));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".costmap_weight", rclcpp::ParameterValue(3.0));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".lethal_cost_threshold", rclcpp::ParameterValue(253));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".unknown_is_obstacle", rclcpp::ParameterValue(true));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".goal_search_radius_m", rclcpp::ParameterValue(0.50));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".max_iterations", rclcpp::ParameterValue(1000000));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".shortcut_path", rclcpp::ParameterValue(true));
  nav2_util::declare_parameter_if_not_declared(
    node_, name_ + ".resample_step_m", rclcpp::ParameterValue(0.05));

  node_->get_parameter(name_ + ".hard_clearance_m", hard_clearance_m_);
  node_->get_parameter(name_ + ".preferred_clearance_m", preferred_clearance_m_);
  node_->get_parameter(name_ + ".clearance_weight", clearance_weight_);
  node_->get_parameter(name_ + ".costmap_weight", costmap_weight_);
  node_->get_parameter(name_ + ".lethal_cost_threshold", lethal_cost_threshold_);
  node_->get_parameter(name_ + ".unknown_is_obstacle", unknown_is_obstacle_);
  node_->get_parameter(name_ + ".goal_search_radius_m", goal_search_radius_m_);
  node_->get_parameter(name_ + ".max_iterations", max_iterations_);
  node_->get_parameter(name_ + ".shortcut_path", shortcut_path_);
  node_->get_parameter(name_ + ".resample_step_m", resample_step_m_);

  if (hard_clearance_m_ < 0.0) {
    throw nav2_core::PlannerException("hard_clearance_m must be non-negative");
  }
  if (preferred_clearance_m_ < hard_clearance_m_ + kEpsilon) {
    preferred_clearance_m_ = hard_clearance_m_ + 0.01;
    RCLCPP_WARN(
      node_->get_logger(),
      "preferred_clearance_m was <= hard_clearance_m; adjusted to %.3f",
      preferred_clearance_m_);
  }
  if (resample_step_m_ <= 0.0) {
    resample_step_m_ = 0.05;
  }
}

nav_msgs::msg::Path ClearanceAStarPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  if (!costmap_) {
    throw nav2_core::PlannerException("ClearanceAStarPlanner has no costmap");
  }

  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*costmap_->getMutex());

  size_x_ = costmap_->getSizeInCellsX();
  size_y_ = costmap_->getSizeInCellsY();
  resolution_ = costmap_->getResolution();

  if (size_x_ == 0 || size_y_ == 0) {
    throw nav2_core::PlannerException("Global costmap is empty");
  }

  unsigned int start_mx = 0;
  unsigned int start_my = 0;
  unsigned int goal_mx = 0;
  unsigned int goal_my = 0;
  worldToMapChecked(start, start_mx, start_my, "start");
  worldToMapChecked(goal, goal_mx, goal_my, "goal");

  const unsigned int start_index = toIndex(start_mx, start_my);
  const unsigned int goal_index = toIndex(goal_mx, goal_my);

  if (isObstacleIndex(start_index)) {
    throw nav2_core::PlannerException("Start pose is inside a lethal or unknown obstacle");
  }

  const auto clearance = computeClearanceField();

  unsigned int safe_goal_index = goal_index;
  geometry_msgs::msg::PoseStamped safe_goal = goal;
  if (isObstacleIndex(goal_index) || clearance[goal_index] + kEpsilon < hard_clearance_m_) {
    if (!findSafeGoal(goal_index, goal, clearance, safe_goal_index, safe_goal)) {
      throw nav2_core::PlannerException("No safe goal found within goal_search_radius_m");
    }
    RCLCPP_WARN(
      node_->get_logger(),
      "Goal was too close to obstacle; projected to safe cell at (%.3f, %.3f)",
      safe_goal.pose.position.x, safe_goal.pose.position.y);
  }

  const auto raw_path = runAStar(start_index, safe_goal_index, clearance);
  if (raw_path.empty()) {
    throw nav2_core::PlannerException("ClearanceAStarPlanner failed to find a valid path");
  }

  return buildPathMsg(raw_path, start, safe_goal, clearance);
}

bool ClearanceAStarPlanner::worldToMapChecked(
  const geometry_msgs::msg::PoseStamped & pose,
  unsigned int & mx,
  unsigned int & my,
  const std::string & label) const
{
  if (!pose.header.frame_id.empty() && pose.header.frame_id != global_frame_) {
    RCLCPP_WARN_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000,
      "%s pose frame is %s, expected %s",
      label.c_str(), pose.header.frame_id.c_str(), global_frame_.c_str());
  }

  if (!costmap_->worldToMap(pose.pose.position.x, pose.pose.position.y, mx, my)) {
    throw nav2_core::PlannerException(label + " pose is outside the global costmap");
  }
  return true;
}

unsigned int ClearanceAStarPlanner::toIndex(unsigned int mx, unsigned int my) const
{
  return my * size_x_ + mx;
}

std::pair<unsigned int, unsigned int> ClearanceAStarPlanner::toCell(unsigned int index) const
{
  return {index % size_x_, index / size_x_};
}

bool ClearanceAStarPlanner::isObstacleCost(unsigned char cost) const
{
  if (cost == nav2_costmap_2d::NO_INFORMATION) {
    return unknown_is_obstacle_;
  }
  return static_cast<int>(cost) >= lethal_cost_threshold_;
}

bool ClearanceAStarPlanner::isObstacleIndex(unsigned int index) const
{
  const auto [mx, my] = toCell(index);
  return isObstacleCost(costmap_->getCost(mx, my));
}

std::vector<double> ClearanceAStarPlanner::computeClearanceField() const
{
  const unsigned int total_cells = size_x_ * size_y_;
  std::vector<double> clearance(total_cells, std::numeric_limits<double>::infinity());
  std::priority_queue<QueueEntry> queue;

  for (unsigned int y = 0; y < size_y_; ++y) {
    for (unsigned int x = 0; x < size_x_; ++x) {
      const unsigned int index = toIndex(x, y);
      if (isObstacleCost(costmap_->getCost(x, y))) {
        clearance[index] = 0.0;
        queue.push({index, 0.0, 0.0});
      }
    }
  }

  while (!queue.empty()) {
    const auto current = queue.top();
    queue.pop();
    if (current.g > clearance[current.index] + kEpsilon) {
      continue;
    }

    const auto [cx_u, cy_u] = toCell(current.index);
    const int cx = static_cast<int>(cx_u);
    const int cy = static_cast<int>(cy_u);

    for (int n = 0; n < kNeighborCount; ++n) {
      const int nx = cx + kDx[n];
      const int ny = cy + kDy[n];
      if (nx < 0 || ny < 0 || nx >= static_cast<int>(size_x_) || ny >= static_cast<int>(size_y_)) {
        continue;
      }

      const unsigned int next_index = toIndex(static_cast<unsigned int>(nx), static_cast<unsigned int>(ny));
      const double step = (kDx[n] == 0 || kDy[n] == 0) ? resolution_ : resolution_ * kSqrt2;
      const double next_distance = current.g + step;
      if (next_distance + kEpsilon < clearance[next_index]) {
        clearance[next_index] = next_distance;
        queue.push({next_index, next_distance, next_distance});
      }
    }
  }

  return clearance;
}

bool ClearanceAStarPlanner::findSafeGoal(
  unsigned int original_goal_index,
  const geometry_msgs::msg::PoseStamped & original_goal,
  const std::vector<double> & clearance,
  unsigned int & safe_goal_index,
  geometry_msgs::msg::PoseStamped & safe_goal) const
{
  const auto [goal_mx_u, goal_my_u] = toCell(original_goal_index);
  const int goal_mx = static_cast<int>(goal_mx_u);
  const int goal_my = static_cast<int>(goal_my_u);
  const int radius_cells = static_cast<int>(std::ceil(goal_search_radius_m_ / resolution_));

  bool found = false;
  double best_score = std::numeric_limits<double>::infinity();
  unsigned int best_index = original_goal_index;

  for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
    for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
      const int mx = goal_mx + dx;
      const int my = goal_my + dy;
      if (mx < 0 || my < 0 || mx >= static_cast<int>(size_x_) || my >= static_cast<int>(size_y_)) {
        continue;
      }

      const double offset_distance = std::hypot(dx * resolution_, dy * resolution_);
      if (offset_distance > goal_search_radius_m_ + kEpsilon) {
        continue;
      }

      const unsigned int index = toIndex(static_cast<unsigned int>(mx), static_cast<unsigned int>(my));
      if (isObstacleIndex(index) || clearance[index] + kEpsilon < hard_clearance_m_) {
        continue;
      }

      const unsigned char cost = costmap_->getCost(static_cast<unsigned int>(mx), static_cast<unsigned int>(my));
      const double cost_term = static_cast<double>(std::min<unsigned char>(
        cost, nav2_costmap_2d::MAX_NON_OBSTACLE)) / nav2_costmap_2d::MAX_NON_OBSTACLE;
      const double score = offset_distance + 0.02 * cost_term - 0.001 * clearance[index];
      if (score < best_score) {
        found = true;
        best_score = score;
        best_index = index;
      }
    }
  }

  if (!found) {
    return false;
  }

  double wx = 0.0;
  double wy = 0.0;
  const auto [best_mx, best_my] = toCell(best_index);
  costmap_->mapToWorld(best_mx, best_my, wx, wy);
  safe_goal_index = best_index;
  safe_goal = original_goal;
  safe_goal.pose.position.x = wx;
  safe_goal.pose.position.y = wy;
  return true;
}

bool ClearanceAStarPlanner::isCellTraversable(
  unsigned int index,
  unsigned int start_index,
  double start_clearance,
  const std::vector<double> & clearance) const
{
  if (isObstacleIndex(index)) {
    return false;
  }
  if (clearance[index] + kEpsilon >= hard_clearance_m_) {
    return true;
  }
  if (index == start_index) {
    return true;
  }

  if (start_clearance + kEpsilon < hard_clearance_m_) {
    const auto [sx_u, sy_u] = toCell(start_index);
    const auto [mx_u, my_u] = toCell(index);
    const double distance_from_start = std::hypot(
      static_cast<double>(static_cast<int>(mx_u) - static_cast<int>(sx_u)) * resolution_,
      static_cast<double>(static_cast<int>(my_u) - static_cast<int>(sy_u)) * resolution_);
    if (distance_from_start <= hard_clearance_m_ + resolution_ &&
      clearance[index] + resolution_ * 0.25 >= start_clearance)
    {
      return true;
    }
  }

  return false;
}

std::vector<unsigned int> ClearanceAStarPlanner::runAStar(
  unsigned int start_index,
  unsigned int goal_index,
  const std::vector<double> & clearance) const
{
  if (start_index == goal_index) {
    return {start_index};
  }

  const unsigned int total_cells = size_x_ * size_y_;
  std::vector<SearchState> states(total_cells);
  for (auto & state : states) {
    state.g = std::numeric_limits<double>::infinity();
    state.f = std::numeric_limits<double>::infinity();
  }

  std::priority_queue<QueueEntry> open;
  const double start_clearance = clearance[start_index];
  states[start_index].g = 0.0;
  states[start_index].f = heuristic(start_index, goal_index);
  states[start_index].opened = true;
  open.push({start_index, states[start_index].f, 0.0});

  int iterations = 0;
  while (!open.empty()) {
    if (++iterations > max_iterations_) {
      RCLCPP_WARN(node_->get_logger(), "A* reached max_iterations=%d", max_iterations_);
      break;
    }

    const auto current = open.top();
    open.pop();
    if (states[current.index].closed) {
      continue;
    }
    if (current.g > states[current.index].g + kEpsilon) {
      continue;
    }

    states[current.index].closed = true;
    if (current.index == goal_index) {
      return reconstructPath(start_index, goal_index, states);
    }

    const auto [cx_u, cy_u] = toCell(current.index);
    const int cx = static_cast<int>(cx_u);
    const int cy = static_cast<int>(cy_u);

    for (int n = 0; n < kNeighborCount; ++n) {
      const int nx = cx + kDx[n];
      const int ny = cy + kDy[n];
      if (nx < 0 || ny < 0 || nx >= static_cast<int>(size_x_) || ny >= static_cast<int>(size_y_)) {
        continue;
      }

      const unsigned int next_index = toIndex(static_cast<unsigned int>(nx), static_cast<unsigned int>(ny));
      if (states[next_index].closed ||
        !isCellTraversable(next_index, start_index, start_clearance, clearance))
      {
        continue;
      }

      const double next_g = states[current.index].g + traversalCost(current.index, next_index, clearance);
      if (!states[next_index].opened || next_g + kEpsilon < states[next_index].g) {
        states[next_index].opened = true;
        states[next_index].parent = static_cast<int>(current.index);
        states[next_index].g = next_g;
        states[next_index].f = next_g + heuristic(next_index, goal_index);
        open.push({next_index, states[next_index].f, next_g});
      }
    }
  }

  return {};
}

std::vector<unsigned int> ClearanceAStarPlanner::reconstructPath(
  unsigned int start_index,
  unsigned int goal_index,
  const std::vector<SearchState> & states) const
{
  std::vector<unsigned int> path;
  int current = static_cast<int>(goal_index);
  while (current >= 0) {
    path.push_back(static_cast<unsigned int>(current));
    if (static_cast<unsigned int>(current) == start_index) {
      std::reverse(path.begin(), path.end());
      return path;
    }
    current = states[static_cast<unsigned int>(current)].parent;
  }
  return {};
}

double ClearanceAStarPlanner::heuristic(unsigned int from_index, unsigned int to_index) const
{
  const auto [fx_u, fy_u] = toCell(from_index);
  const auto [tx_u, ty_u] = toCell(to_index);
  return std::hypot(
    static_cast<double>(static_cast<int>(fx_u) - static_cast<int>(tx_u)) * resolution_,
    static_cast<double>(static_cast<int>(fy_u) - static_cast<int>(ty_u)) * resolution_);
}

double ClearanceAStarPlanner::traversalCost(
  unsigned int from_index,
  unsigned int to_index,
  const std::vector<double> & clearance) const
{
  const double step = heuristic(from_index, to_index);
  const auto [mx, my] = toCell(to_index);
  const unsigned char raw_cost = costmap_->getCost(mx, my);
  const double cost_norm = static_cast<double>(
    std::min<unsigned char>(raw_cost, nav2_costmap_2d::MAX_NON_OBSTACLE)) /
    nav2_costmap_2d::MAX_NON_OBSTACLE;

  double clearance_penalty = 0.0;
  const double distance = clearance[to_index];
  if (std::isfinite(distance) && distance + kEpsilon < preferred_clearance_m_) {
    double ratio = 0.0;
    if (distance + kEpsilon < hard_clearance_m_) {
      ratio = 1.0 + (hard_clearance_m_ - distance) / std::max(hard_clearance_m_, 0.01);
    } else {
      ratio = (preferred_clearance_m_ - distance) /
        std::max(preferred_clearance_m_ - hard_clearance_m_, 0.01);
    }
    clearance_penalty = clearance_weight_ * ratio * ratio;
  }

  return step * (1.0 + costmap_weight_ * cost_norm + clearance_penalty);
}

bool ClearanceAStarPlanner::lineIsSafe(
  unsigned int from_index,
  unsigned int to_index,
  const std::vector<double> & clearance) const
{
  const auto [x0_u, y0_u] = toCell(from_index);
  const auto [x1_u, y1_u] = toCell(to_index);
  int x0 = static_cast<int>(x0_u);
  int y0 = static_cast<int>(y0_u);
  const int x1 = static_cast<int>(x1_u);
  const int y1 = static_cast<int>(y1_u);

  const int dx = std::abs(x1 - x0);
  const int sx = x0 < x1 ? 1 : -1;
  const int dy = -std::abs(y1 - y0);
  const int sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;

  while (true) {
    const unsigned int index = toIndex(static_cast<unsigned int>(x0), static_cast<unsigned int>(y0));
    if (isObstacleIndex(index) || clearance[index] + kEpsilon < hard_clearance_m_) {
      return false;
    }
    if (x0 == x1 && y0 == y1) {
      return true;
    }
    const int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
}

std::vector<unsigned int> ClearanceAStarPlanner::shortcutPath(
  const std::vector<unsigned int> & raw_path,
  const std::vector<double> & clearance) const
{
  if (!shortcut_path_ || raw_path.size() <= 2) {
    return raw_path;
  }

  std::vector<unsigned int> result;
  result.reserve(raw_path.size());
  result.push_back(raw_path.front());

  std::size_t i = 0;
  while (i + 1 < raw_path.size()) {
    if (clearance[raw_path[i]] + kEpsilon < hard_clearance_m_) {
      ++i;
      result.push_back(raw_path[i]);
      continue;
    }

    std::size_t best = i + 1;
    for (std::size_t j = raw_path.size() - 1; j > i + 1; --j) {
      if (clearance[raw_path[j]] + kEpsilon < hard_clearance_m_) {
        continue;
      }
      if (lineIsSafe(raw_path[i], raw_path[j], clearance)) {
        best = j;
        break;
      }
    }

    if (result.back() != raw_path[best]) {
      result.push_back(raw_path[best]);
    }
    i = best;
  }

  return result;
}

nav_msgs::msg::Path ClearanceAStarPlanner::buildPathMsg(
  const std::vector<unsigned int> & path_indices,
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal,
  const std::vector<double> & clearance) const
{
  const auto compact_path = shortcutPath(path_indices, clearance);

  std::vector<std::pair<double, double>> control_points;
  control_points.reserve(compact_path.size());
  for (const auto index : compact_path) {
    const auto [mx, my] = toCell(index);
    double wx = 0.0;
    double wy = 0.0;
    costmap_->mapToWorld(mx, my, wx, wy);
    control_points.emplace_back(wx, wy);
  }

  if (!control_points.empty()) {
    control_points.front() = {start.pose.position.x, start.pose.position.y};
    control_points.back() = {goal.pose.position.x, goal.pose.position.y};
  }

  std::vector<std::pair<double, double>> samples;
  if (!control_points.empty()) {
    samples.push_back(control_points.front());
  }

  for (std::size_t i = 0; i + 1 < control_points.size(); ++i) {
    const auto [x0, y0] = control_points[i];
    const auto [x1, y1] = control_points[i + 1];
    const double length = std::hypot(x1 - x0, y1 - y0);
    const int steps = std::max(1, static_cast<int>(std::ceil(length / resample_step_m_)));
    for (int s = 1; s <= steps; ++s) {
      const double t = static_cast<double>(s) / static_cast<double>(steps);
      samples.emplace_back(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
    }
  }

  nav_msgs::msg::Path path;
  path.header.frame_id = global_frame_;
  path.header.stamp = node_->now();
  path.poses.reserve(samples.size());

  for (std::size_t i = 0; i < samples.size(); ++i) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = samples[i].first;
    pose.pose.position.y = samples[i].second;
    pose.pose.position.z = 0.0;

    if (i + 1 < samples.size()) {
      const double yaw = std::atan2(samples[i + 1].second - samples[i].second,
        samples[i + 1].first - samples[i].first);
      pose.pose.orientation = yawToQuaternion(yaw);
    } else {
      pose.pose.orientation = goal.pose.orientation;
    }
    path.poses.push_back(pose);
  }

  if (path.poses.empty()) {
    geometry_msgs::msg::PoseStamped pose = start;
    pose.header = path.header;
    path.poses.push_back(pose);
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "Clearance path created: raw=%zu compact=%zu samples=%zu",
    path_indices.size(), compact_path.size(), path.poses.size());

  return path;
}

geometry_msgs::msg::Quaternion ClearanceAStarPlanner::yawToQuaternion(double yaw) const
{
  geometry_msgs::msg::Quaternion q;
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw * 0.5);
  q.w = std::cos(yaw * 0.5);
  return q;
}

}  // namespace wl100_clearance_planner

PLUGINLIB_EXPORT_CLASS(
  wl100_clearance_planner::ClearanceAStarPlanner,
  nav2_core::GlobalPlanner)
