#include <covariance.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

double kernel_weight(Kernel kernel, double t) {
    t = std::abs(t);
    if (!(t < 1.0)) return 0.0;  // also catches NaN
    const double s = 1.0 - t * t;
    switch (kernel) {
        case Kernel::Triweight:
            return s * s * s;
        case Kernel::Bump:
            return std::exp(1.0 - 1.0 / s);
        case Kernel::Dumb:
            double threshold = 0.9;
            if (t < threshold) {
                return 1.0;
            }
            else {
                double t = (1.0 - t) / (1.0 - threshold);
                return t;
            };
    }
    return 0.0;
}

namespace {

struct LocalFit {
    Eigen::MatrixXd P;   // trace-normalized scatter
    Eigen::VectorXd mu;  // location
    int iterations = 0;
    bool converged = false;
};

// Fixed-point iterations for one neighborhood.
//   Y   : m x n coordinates of the neighborhood points (including x itself)
//   phi : m kernel weights
//   x   : query point
//   r   : kernel radius (only used to set a floor on the Mahalanobis-like distances)
LocalFit fit_local(const Eigen::MatrixXd& Y, const Eigen::VectorXd& phi, const Eigen::VectorXd& x,
                   double r, const TylerParams& params) {
    const Eigen::Index m = Y.rows();
    const Eigen::Index n = Y.cols();
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);

    LocalFit fit;
    fit.P = I / static_cast<double>(n);
    fit.mu = params.joint_location ? Eigen::VectorXd((Y.transpose() * phi) / phi.sum()) : x;

    // d_j >= ||x_j - mu|| for a trace-normalized P, so a floor relative to r is safe
    // and prevents division by zero when a point coincides with mu.
    const double d_floor = 1e-8 * r;

    Eigen::MatrixXd D(m, n);
    for (int it = 0; it < params.max_iter; ++it) {
        // MAYBE this should be Y - x
        D = Y.rowwise() - fit.mu.transpose();

        // Mahalanobis-like distances d_j = sqrt(D_j^T P^{-1} D_j) through a Cholesky factor of P.
        Eigen::LLT<Eigen::MatrixXd> llt(fit.P);
        if (llt.info() != Eigen::Success) {
            fit.iterations = it;
            return fit;  // not converged
        }
        const Eigen::MatrixXd Z = llt.matrixL().solve(D.transpose());  // n x m
        const Eigen::VectorXd d =
            Z.colwise().squaredNorm().transpose().cwiseSqrt().cwiseMax(d_floor);

        // Scatter update (uses mu^{(k-1)} and d^{(k-1)}).
        const Eigen::VectorXd w = phi.array() / d.array().square();
        Eigen::MatrixXd P_new = D.transpose() * w.asDiagonal() * D;
        P_new = 0.5 * (P_new + P_new.transpose());
        P_new /= P_new.trace();
        P_new += params.ridge * I;
        P_new /= P_new.trace();

        // Location update (weighted spatial-median step).
        if (params.joint_location) {
            const Eigen::VectorXd v = phi.array() / d.array();
            fit.mu = (Y.transpose() * v) / v.sum();
        }

        const double change = (P_new - fit.P).norm();
        fit.P = P_new;
        fit.iterations = it + 1;
        if (change < params.tol) {
            fit.converged = true;
            return fit;
        }
    }
    return fit;
}

}  // namespace

TylerResult tyler_covariances(const Eigen::Ref<const RowMatrixXd>& points,
                              const TylerParams& params) {
    const Eigen::Index N = points.rows();
    const Eigen::Index n = points.cols();
    const int k = params.k;

    if (k < n + 1) {
        throw std::invalid_argument("tyler_covariances: k must be >= dim + 1");
    }
    if (N < k + 2) {
        throw std::invalid_argument("tyler_covariances: need at least k + 2 points");
    }

    // k + 1 neighbors (self excluded): the first k are used, the (k+1)-th defines the radius.
    const KNN::KnnResult nn = KNN::knn(points, k + 1, /*include_self=*/false);

    TylerResult out;
    out.covariances.resize(N, n * n);
    out.locations.resize(N, n);
    out.radii.resize(N);
    out.iterations.resize(N);
    out.converged.resize(N);

    // No exceptions may escape the parallel region: everything below is non-throwing.
#pragma omp parallel
    {
        Eigen::MatrixXd Y(k + 1, n);
        Eigen::VectorXd phi(k + 1);

#pragma omp for schedule(dynamic, 64)
        for (Eigen::Index i = 0; i < N; ++i) {
            const Eigen::VectorXd x = points.row(i).transpose();
            const double r = nn.distances(i, k);

            LocalFit fit;
            if (!(r > 0.0)) {
                // Degenerate neighborhood (k + 1 duplicates of x): isotropic fallback.
                fit.P = Eigen::MatrixXd::Identity(n, n) / static_cast<double>(n);
                fit.mu = x;
            } else {
                Y.row(0) = x.transpose();
                phi(0) = 1.0;  // K(0) = 1 for both kernels
                for (int j = 0; j < k; ++j) {
                    Y.row(j + 1) = points.row(nn.indices(i, j));
                    phi(j + 1) = kernel_weight(params.kernel, nn.distances(i, j) / r);
                }
                fit = fit_local(Y, phi, x, r, params);
            }

            Eigen::Map<RowMatrixXd>(out.covariances.row(i).data(), n, n) = fit.P;
            out.locations.row(i) = fit.mu.transpose();
            out.radii(i) = r;
            out.iterations(i) = fit.iterations;
            out.converged(i) = fit.converged ? 1 : 0;
        }
    }
    return out;
}

}