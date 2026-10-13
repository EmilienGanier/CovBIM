#pragma once

#include <Eigen/Dense>

namespace KNN {

/// Row-major dense matrices: one point per row, same memory layout as a C-contiguous numpy array.
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// k-nearest-neighbor result: row i holds the neighbors of point i, sorted by increasing distance.
struct KnnResult {
    RowMatrixXi indices;    // N x k
    RowMatrixXd distances;  // N x k, Euclidean distances (not squared)
};

/// Exact k nearest neighbors of every point in `points` (N x dim, one point per row).
/// If `include_self` is false (default), a point is not its own neighbor.
/// Throws std::invalid_argument if k < 1 or there are not enough points.
KnnResult knn(const Eigen::Ref<const RowMatrixXd>& points, int k, bool include_self = false);

}