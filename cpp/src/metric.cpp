#include <metric.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <spd.hpp>

namespace METRIC {

EdgeWeighter::EdgeWeighter(const Eigen::Ref<const KNN::RowMatrixXd>& points,
                           const Eigen::Ref<const KNN::RowMatrixXd>& covariances,
                           const MetricParams& params)
    : params_(params), points_(points) {
    const Eigen::Index N = points.rows();
    const Eigen::Index n = points.cols();

    if (covariances.rows() != N || covariances.cols() != n * n) {
        throw std::invalid_argument("EdgeWeighter: covariances must have shape (N, n*n)");
    }
    if (params.alpha < 0.0 || params.beta < 1.0 || params.lambda < 0.0 || !(params.min_eig > 0.0)) {
        throw std::invalid_argument("EdgeWeighter: need alpha >= 0, beta >= 1, lambda >= 0, min_eig > 0");
    }
    // The determinant penalty is defined on trace-normalized matrices. Checks that the input covariances are trace-normalized.
    for (Eigen::Index i = 0; i < N; ++i) {
        const Eigen::Map<const KNN::RowMatrixXd> P(covariances.row(i).data(), n, n);
        if (std::abs(P.trace() - 1.0) > 1e-6) {
            throw std::invalid_argument("EdgeWeighter: covariances must be trace-normalized");
        }
    }

    const bool use_logs = (params.spd_metric == SpdMetric::LogEuclidean);
    logdet_.resize(N);
    feats_.resize(N, n * n);
    if (!use_logs) feats_ = covariances;

#pragma omp parallel
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(n);  // one solver per thread

#pragma omp for schedule(static)
        for (Eigen::Index i = 0; i < N; ++i) {
            const Eigen::Map<const KNN::RowMatrixXd> P(covariances.row(i).data(), n, n);
            solver.compute(P);
            const Eigen::ArrayXd logs = solver.eigenvalues().array().max(params.min_eig).log();
            logdet_(i) = logs.sum();
            if (use_logs) {
                const Eigen::MatrixXd L =
                    solver.eigenvectors() * logs.matrix().asDiagonal() * solver.eigenvectors().transpose();
                Eigen::Map<KNN::RowMatrixXd>(feats_.row(i).data(), n, n) = L;
            }
        }
    }
}

double EdgeWeighter::weight(Eigen::Index i, Eigen::Index j, double beta) const {
    const Eigen::Index n = points_.cols();

    double d_spd = 0.0;
    switch (params_.spd_metric) {
        //I could also use the function from spd.hpp directly
        case SpdMetric::Frobenius:
        case SpdMetric::LogEuclidean:
            d_spd = (feats_.row(i) - feats_.row(j)).norm();
            break;
        // this last case technically works but it would be more intuitive to plug the covariance matrices directly (which i the case here since feats_ is log(P_i) for Log-Euclidean and P_i otherwise)
        case SpdMetric::Airm: {
            const Eigen::MatrixXd Pi = Eigen::Map<const KNN::RowMatrixXd>(feats_.row(i).data(), n, n);
            const Eigen::MatrixXd Pj = Eigen::Map<const KNN::RowMatrixXd>(feats_.row(j).data(), n, n);
            d_spd = SPD::d_AIRM(Pi, Pj, params_.min_eig);
            break;
        }
    }
    d_spd *= params_.lambda;

    const double dx2 = (points_.row(i) - points_.row(j)).squaredNorm();
    const double d_prod = std::sqrt(dx2 + d_spd * d_spd);

    // max(det_i, det_j)^alpha, computed through log-determinants for numerical safety.
    const double penalty = std::exp(params_.alpha * std::max(logdet_(i), logdet_(j)));
    return penalty * std::pow(d_prod, beta);
}


// create a graph from the neighbor indices, with weights given by the EdgeWeighter (spd metric).
GRAPH::CsrGraph build_graph(const EdgeWeighter& weighter,
                     const Eigen::Ref<const KNN::RowMatrixXi>& neighbor_indices) {
    const Eigen::Index N = neighbor_indices.rows();
    const Eigen::Index k = neighbor_indices.cols();
    if (N != weighter.num_points()) {
        throw std::invalid_argument("build_graph: neighbor_indices and weighter disagree on N");
    }

    // Both directions of every neighbor relation, then sort + unique -> symmetric, rows sorted.
    std::vector<std::pair<Eigen::Index, Eigen::Index>> edges;
    edges.reserve(static_cast<size_t>(2 * N * k));
    for (Eigen::Index i = 0; i < N; ++i) {
        for (Eigen::Index c = 0; c < k; ++c) {
            const Eigen::Index j = neighbor_indices(i, c);
            if (j < 0 || j >= N) throw std::invalid_argument("build_graph: neighbor index out of range");
            if (j == i) continue;
            edges.emplace_back(i, j);
            edges.emplace_back(j, i);
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    // now create the CSR graph
    const Eigen::Index nnz = static_cast<Eigen::Index>(edges.size());
    GRAPH::CsrGraph g;
    g.n_nodes = N;
    g.indptr = GRAPH::IndexVector::Zero(N + 1);
    g.indices.resize(nnz);
    g.weights.resize(nnz);

    for (const auto& e : edges) ++g.indptr(e.first + 1);
    for (Eigen::Index i = 0; i < N; ++i) g.indptr(i + 1) += g.indptr(i);

#pragma omp parallel for schedule(static)
    for (Eigen::Index e = 0; e < nnz; ++e) {
        g.indices(e) = edges[e].second;
        g.weights(e) = weighter(edges[e].first, edges[e].second);
    }
    return g;
}

// see metric.hpp for documentation
Eigen::VectorXd tree_lengths(const EdgeWeighter& weighter, const GRAPH::IndexVector& predecessors,
                             const GRAPH::IndexVector& order, Eigen::Index source, double beta) {
    const Eigen::Index N = weighter.num_points();
    if (predecessors.size() != N) {
        throw std::invalid_argument("tree_lengths: predecessors must have one entry per point");
    }
    if (source < 0 || source >= N) throw std::invalid_argument("tree_lengths: source out of range");

    Eigen::VectorXd lengths = Eigen::VectorXd::Constant(N, std::numeric_limits<double>::infinity());
    lengths(source) = 0.0;

    // `order` lists nodes by settling time, so each predecessor is processed before its children.
    // we could also use a recursive function without the need for the order
    for (Eigen::Index t = 0; t < order.size(); ++t) {
        const Eigen::Index v = order(t);
        if (v == source) continue;
        const Eigen::Index u = predecessors(v);
        if (u < 0) continue;  // not reachable
        lengths(v) = lengths(u) + weighter.weight(u, v, beta);
    }
    return lengths;
}

}