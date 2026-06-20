#include <hdl_global_localization/bbs/bbs_localization.hpp>
#include <hdl_global_localization/bbs/occupancy_gridmap.hpp>

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace hdl_global_localization {

namespace {
constexpr double kPi = 3.14159265358979323846;

double normalize_angle(double angle) {
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

struct BucketKey {
  int x;
  int y;
  int yaw;

  bool operator==(const BucketKey& other) const {
    return x == other.x && y == other.y && yaw == other.yaw;
  }
};

struct BucketKeyHash {
  std::size_t operator()(const BucketKey& key) const {
    std::size_t seed = 0;
    seed ^= std::hash<int>()(key.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>()(key.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>()(key.yaw) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};
}

DiscreteTransformation::DiscreteTransformation() {
  this->level = -1;
}

DiscreteTransformation::DiscreteTransformation(int level, int x, int y, int theta) {
  this->level = level;
  this->x = x;
  this->y = y;
  this->theta = theta;
  this->score = 0.0;
}

DiscreteTransformation::~DiscreteTransformation() {}

bool DiscreteTransformation::operator<(const DiscreteTransformation& rhs) const {
  return score < rhs.score;
}

bool DiscreteTransformation::is_leaf() const {
  return level == 0;
}

Eigen::Isometry2f DiscreteTransformation::transformation(double theta_resolution, const std::vector<std::shared_ptr<OccupancyGridMap>>& gridmap_pyramid) const {
  double trans_resolution = gridmap_pyramid[level]->grid_resolution();

  Eigen::Isometry2f trans = Eigen::Isometry2f::Identity();
  trans.linear() = Eigen::Rotation2Df(theta_resolution * theta).toRotationMatrix();
  trans.translation() = Eigen::Vector2f(trans_resolution * x, trans_resolution * y);

  return trans;
}

DiscreteTransformation::Points DiscreteTransformation::transform(const Points& points, double trans_resolution, double theta_resolution) {
  Points transformed(points.size());

  Eigen::Map<const Eigen::Matrix<float, 2, -1>> src(points[0].data(), 2, points.size());
  Eigen::Map<Eigen::Matrix<float, 2, -1>> dst(transformed[0].data(), 2, points.size());

  Eigen::Matrix2f rot = Eigen::Rotation2Df(theta_resolution * theta).toRotationMatrix();
  Eigen::Vector2f trans(trans_resolution * x, trans_resolution * y);

  dst = (rot * src).colwise() + trans;

  return transformed;
}

double DiscreteTransformation::calc_score(const Points& points, double theta_resolution, const std::vector<std::shared_ptr<OccupancyGridMap>>& gridmap_pyramid) {
  const auto& gridmap = gridmap_pyramid[level];
  auto transformed = transform(points, gridmap->grid_resolution(), theta_resolution);
  score = gridmap->calc_score(transformed);
  return score;
}

std::vector<DiscreteTransformation> DiscreteTransformation::branch() {
  std::vector<DiscreteTransformation> b;
  b.reserve(4);
  b.emplace_back(DiscreteTransformation(level - 1, x * 2, y * 2, theta));
  b.emplace_back(DiscreteTransformation(level - 1, x * 2 + 1, y * 2, theta));
  b.emplace_back(DiscreteTransformation(level - 1, x * 2, y * 2 + 1, theta));
  b.emplace_back(DiscreteTransformation(level - 1, x * 2 + 1, y * 2 + 1, theta));
  return b;
}

BBSLocalization::BBSLocalization(const BBSParams& params) : params(params) {}

BBSLocalization::~BBSLocalization() {}

void BBSLocalization::set_map(const BBSLocalization::Points& map_points, double resolution, int width, int height, int pyramid_levels, int max_points_per_cell) {
  gridmap_pyramid.resize(pyramid_levels);
  gridmap_pyramid[0].reset(new OccupancyGridMap(resolution, width, height));
  gridmap_pyramid[0]->insert_points(map_points, max_points_per_cell);

  for (int i = 1; i < pyramid_levels; i++) {
    gridmap_pyramid[i] = gridmap_pyramid[i - 1]->pyramid_up();
  }
}

boost::optional<Eigen::Isometry2f> BBSLocalization::localize(const BBSLocalization::Points& scan_points, double min_score, double* best_score) {
  auto results = localize_n(scan_points, min_score, 1);

  if (results.empty()) {
    return boost::none;
  }

  if (best_score) {
    *best_score = results.front().score;
  }

  return results.front().pose;
}

BBSLocalization::Results BBSLocalization::localize_n(const BBSLocalization::Points& scan_points, double min_score, int max_num_candidates) {
  Results results;
  max_num_candidates = std::max(1, max_num_candidates);

  theta_resolution = std::acos(1 - std::pow(gridmap_pyramid[0]->grid_resolution(), 2) / (2 * std::pow(params.max_range, 2)));

  double pruning_score = min_score;
  std::vector<DiscreteTransformation> best_transforms;
  best_transforms.reserve(max_num_candidates);

  auto trans_queue = create_init_transset(scan_points);

  // ROS_INFO_STREAM("Branch-and-Bound");
  while (!trans_queue.empty()) {
    // std::cout << trans_queue.size() << std::endl;

    auto trans = trans_queue.top();
    trans_queue.pop();

    if (trans.score < pruning_score) {
      break;
    }

    if (trans.is_leaf()) {
      best_transforms.push_back(trans);
      std::sort(best_transforms.begin(), best_transforms.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.score > rhs.score;
      });

      if (static_cast<int>(best_transforms.size()) > max_num_candidates) {
        best_transforms.resize(max_num_candidates);
      }
      if (static_cast<int>(best_transforms.size()) == max_num_candidates) {
        pruning_score = best_transforms.back().score;
      }
    } else {
      auto children = trans.branch();
      for (auto& child : children) {
        child.calc_score(scan_points, theta_resolution, gridmap_pyramid);
        trans_queue.push(child);
      }
    }
  }

  results.reserve(best_transforms.size());
  for (const auto& trans : best_transforms) {
    results.emplace_back(trans.score, trans.transformation(theta_resolution, gridmap_pyramid));
  }

  return results;
}

BBSLocalization::Results BBSLocalization::localize_n_distributed(
  const BBSLocalization::Points& scan_points,
  double min_score,
  int max_num_candidates,
  double bucket_xy,
  double bucket_yaw,
  int max_leaf_evaluations) {
  Results results;
  max_num_candidates = std::max(1, max_num_candidates);
  bucket_xy = std::max(bucket_xy, static_cast<double>(gridmap_pyramid[0]->grid_resolution()));
  bucket_yaw = std::max(bucket_yaw, 1.0e-3);
  max_leaf_evaluations = std::max(max_num_candidates, max_leaf_evaluations);

  theta_resolution = std::acos(1 - std::pow(gridmap_pyramid[0]->grid_resolution(), 2) / (2 * std::pow(params.max_range, 2)));

  auto trans_queue = create_init_transset(scan_points);
  std::unordered_map<BucketKey, DiscreteTransformation, BucketKeyHash> best_by_bucket;
  int leaf_evaluations = 0;

  while (!trans_queue.empty()) {
    auto trans = trans_queue.top();
    trans_queue.pop();

    if (trans.score < min_score) {
      break;
    }

    if (trans.is_leaf()) {
      leaf_evaluations++;

      const auto pose = trans.transformation(theta_resolution, gridmap_pyramid);
      const double yaw = normalize_angle(theta_resolution * trans.theta);
      const BucketKey key{
        static_cast<int>(std::floor(pose.translation().x() / bucket_xy)),
        static_cast<int>(std::floor(pose.translation().y() / bucket_xy)),
        static_cast<int>(std::floor((yaw + kPi) / bucket_yaw))
      };

      auto found = best_by_bucket.find(key);
      if (found == best_by_bucket.end() || trans.score > found->second.score) {
        best_by_bucket[key] = trans;
      }

      if (leaf_evaluations >= max_leaf_evaluations) {
        break;
      }
      continue;
    }

    auto children = trans.branch();
    for (auto& child : children) {
      child.calc_score(scan_points, theta_resolution, gridmap_pyramid);
      trans_queue.push(child);
    }
  }

  std::vector<DiscreteTransformation> best_transforms;
  best_transforms.reserve(best_by_bucket.size());
  for (const auto& item : best_by_bucket) {
    best_transforms.push_back(item.second);
  }

  std::sort(best_transforms.begin(), best_transforms.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.score > rhs.score;
  });

  if (static_cast<int>(best_transforms.size()) > max_num_candidates) {
    best_transforms.resize(max_num_candidates);
  }

  results.reserve(best_transforms.size());
  for (const auto& trans : best_transforms) {
    results.emplace_back(trans.score, trans.transformation(theta_resolution, gridmap_pyramid));
  }

  return results;
}

std::shared_ptr<const OccupancyGridMap> BBSLocalization::gridmap() const {
  return gridmap_pyramid[0];
}

std::priority_queue<DiscreteTransformation> BBSLocalization::create_init_transset(const Points& scan_points) const {
  double trans_res = gridmap_pyramid.back()->grid_resolution();
  std::pair<int, int> tx_range(std::floor(params.min_tx / trans_res), std::ceil(params.max_tx / trans_res));
  std::pair<int, int> ty_range(std::floor(params.min_ty / trans_res), std::ceil(params.max_ty / trans_res));
  std::pair<int, int> theta_range(std::floor(params.min_theta / theta_resolution), std::ceil(params.max_theta / theta_resolution));

  // ROS_INFO_STREAM("Resolution trans:" << trans_res << " theta:" << theta_resolution);
  // ROS_INFO_STREAM("TransX range:" << tx_range.first << " " << tx_range.second);
  // ROS_INFO_STREAM("TransY range:" << ty_range.first << " " << ty_range.second);
  // ROS_INFO_STREAM("Theta  range:" << theta_range.first << " " << theta_range.second);

  std::vector<DiscreteTransformation> transset;
  transset.reserve((tx_range.second - tx_range.first) * (ty_range.second - ty_range.first) * (theta_range.second - theta_range.first));
  for (int tx = tx_range.first; tx <= tx_range.second; tx++) {
    for (int ty = ty_range.first; ty <= ty_range.second; ty++) {
      for (int theta = theta_range.first; theta <= theta_range.second; theta++) {
        int level = gridmap_pyramid.size() - 1;
        transset.emplace_back(DiscreteTransformation(level, tx, ty, theta));
      }
    }
  }

  // ROS_INFO_STREAM("Initial transformation set size:" << transset.size());

#pragma omp parallel for
  for (int i = 0; i < transset.size(); i++) {
    auto& trans = transset[i];
    const auto& gridmap = gridmap_pyramid[trans.level];
    trans.calc_score(scan_points, theta_resolution, gridmap_pyramid);
  }

  return std::priority_queue<DiscreteTransformation>(transset.begin(), transset.end());
}

}  // namespace  hdl_global_localization
