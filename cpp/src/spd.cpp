#include <spd.hpp>

namespace SPD {

    Eigen::MatrixXd spd_log(const Eigen::MatrixXd& P, double min_eig) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(P);
        Eigen::VectorXd logs = solver.eigenvalues().array().max(min_eig).log();
        return solver.eigenvectors() * logs.asDiagonal() * solver.eigenvectors().transpose();
    }

    double d_LE(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q) {
        return (spd_log(P) - spd_log(Q)).norm();     // check that Eigen's norm() is the Frobenius norm
    }

    double d_AIRM(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q, double min_eig) {
    // Eigenvalues of Q x = lambda P x, i.e. of P^{-1/2} Q P^{-1/2}
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(Q, P, Eigen::EigenvaluesOnly);
    Eigen::ArrayXd logs = solver.eigenvalues().array().max(min_eig).log();
    return std::sqrt(logs.square().sum());       // Frobenius norm of log(...) = l2 norm of log-eigenvalues
}

}