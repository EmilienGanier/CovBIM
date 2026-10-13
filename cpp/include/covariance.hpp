#pragma once

#include <Eigen/Dense>

#include <knn.hpp>

namespace {

/// Row-major dense matrices: one point per row, same memory layout as a C-contiguous numpy array.
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// Smooth, compactly supported kernels K(t) on t = ||x_j - x|| / r, with K(0) = 1 and K(t) = 0 for t >= 1.
enum class Kernel {
    Triweight,  // (1 - t^2)^3         : C^2 on R^n (all derivatives up to order 2 vanish at t = 1)
    Bump,       // exp(1 - 1/(1 - t^2)): C^infinity
    Dumb        // continuous but not C^1
};

/// Evaluates the kernel at t >= 0.
double kernel_weight(Kernel kernel, double t);

struct TylerParams {
    int k = 30;                   // neighborhood size: x itself + its k nearest neighbors get positive weight
    Kernel kernel = Kernel::Dumb;
    bool joint_location = true;   // true: estimate location and scatter; false: location fixed at x
    double tol = 1e-6;            // stop when ||P_new - P||_F < tol
    int max_iter = 500;
    double ridge = 1e-10;         // relative regularization keeping P invertible (P += ridge * I, then re-normalized)
};

struct TylerResult {
    RowMatrixXd covariances;   // N x (n*n): row i is the trace-normalized matrix P_i, row-major (reshape to (N, n, n))
    RowMatrixXd locations;     // N x n: robust location mu_i (equal to x_i if joint_location == false)
    Eigen::VectorXd radii;     // N: kernel radius r_i used at each point
    Eigen::VectorXi iterations;  // N: number of iterations performed
    Eigen::VectorXi converged;   // N: 1 if the tolerance was reached, 0 otherwise
};

/// Localized, kernel-weighted Tyler M-estimator of scatter, computed at every point of `points` (N x n).
/// The radius r_i is the distance from x_i to its (k+1)-th nearest neighbor, so the kernel vanishes exactly
/// there and the estimator varies continuously as points enter or leave the neighborhood.
/// The result is trace-normalized (trace(P_i) = 1).
/// Throws std::invalid_argument if k < n + 1 or if there are fewer than k + 2 points.
TylerResult tyler_covariances(const Eigen::Ref<const RowMatrixXd>& points,
                              const TylerParams& params = {});

}