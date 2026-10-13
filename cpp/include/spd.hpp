#pragma once

#include <Eigen/Dense>

namespace SPD {

    Eigen::MatrixXd spd_log(const Eigen::MatrixXd& P, double min_eig = 1e-12);

    /// Log-Euclidean distance ||log(P) - log(Q)||_F between SPD matrices.
    double d_LE(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q);

    /// Affine-invariant Riemannian distance ||log(P^{-1/2} Q P^{-1/2})||_F between SPD matrices.
    double d_AIRM(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q, double min_eig = 1e-12);

}