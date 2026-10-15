#pragma once

#include <Eigen/Dense>

#include <graph.hpp>
#include <knn.hpp>  // RowMatrixXd, RowMatrixXi

namespace METRIC {

enum class SpdMetric { Frobenius, LogEuclidean, Airm };

struct MetricParams {
    double alpha = 1.0;    // exponent of the determinant penalty (0 disables it)
    double beta = 1.0;     // exponent applied to the product-metric length of each edge
    double lambda = 1.0;   // stiffness: d_SPD = lambda * ||.||
    SpdMetric spd_metric = SpdMetric::LogEuclidean;
    double min_eig = 1e-12;  // eigenvalue floor used in logarithms
};

/// Evaluates the edge weight
///     w(i, j) = max(det P_i, det P_j)^alpha * d_M((x_i, P_i), (x_j, P_j))^beta,
///     d_M^2   = ||x_i - x_j||^2 + (lambda * d_SPD(P_i, P_j))^2,
/// where P_i are trace-normalized SPD matrices (as returned by tyler_covariances).
/// Per-point quantities (matrix logarithms, log-determinants) are computed once in the
/// constructor; the object owns copies of its data and can be used from several threads.
class EdgeWeighter {
public:
    /// points: N x n. covariances: N x n^2, row i = trace-normalized P_i flattened (row-major).
    /// Throws std::invalid_argument on inconsistent sizes, invalid parameters or non-normalized matrices.
    EdgeWeighter(const Eigen::Ref<const KNN::RowMatrixXd>& points,
                 const Eigen::Ref<const KNN::RowMatrixXd>& covariances,
                 const MetricParams& params = {});

    Eigen::Index num_points() const { return points_.rows(); }
    const MetricParams& params() const { return params_; }

    /// Weight of the edge (i, j) with the exponent beta given in the parameters.
    double operator()(Eigen::Index i, Eigen::Index j) const { return weight(i, j, params_.beta); }

    /// Weight of the edge (i, j) with an explicit exponent beta (e.g. beta = 1 for the Riemannian length).
    double weight(Eigen::Index i, Eigen::Index j, double beta) const;

private:
    MetricParams params_;
    KNN::RowMatrixXd points_;     // N x n
    KNN::RowMatrixXd feats_;      // N x n^2: log(P_i) for Log-Euclidean, P_i otherwise
    Eigen::VectorXd logdet_;  // N: log det P_i
};

/// Builds the symmetrized neighbor graph: edge {i, j} exists if j is among the neighbors of i or vice versa.
/// neighbor_indices: N x k (e.g. KnnResult::indices). Self-loops are ignored.
GRAPH::CsrGraph build_graph(const EdgeWeighter& weighter,
                     const Eigen::Ref<const KNN::RowMatrixXi>& neighbor_indices);

/// Length of the tree paths from `source` under the exponent `beta`: lengths[v] is the sum of
/// weighter.weight(., ., beta) along the path source -> v in the predecessor tree.
/// Use the predecessors and settling order returned by dijkstra(); unreachable nodes get +inf.
/// With beta = 1 this gives the Riemannian length of the paths selected with beta > 1.
Eigen::VectorXd tree_lengths(const EdgeWeighter& weighter, const GRAPH::IndexVector& predecessors,
                             const GRAPH::IndexVector& order, Eigen::Index source, double beta = 1.0);

}