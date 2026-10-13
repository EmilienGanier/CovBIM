#include <spd.hpp>

namespace SPD {

    Eigen::MatrixXd spd_log(const Eigen::MatrixXd& P, double min_eig) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(P);
        Eigen::VectorXd logs = solver.eigenvalues().array().max(min_eig).log();
        return solver.eigenvectors() * logs.asDiagonal() * solver.eigenvectors().transpose();
    }

    double log_euclidean_distance(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q) {
        return (spd_log(P) - spd_log(Q)).norm();     // check that Eigen's norm() is the Frobenius norm
    }

}