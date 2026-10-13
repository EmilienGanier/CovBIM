// bindings/module.cpp
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
// #include <geodsd/covariance.hpp>
// #include <test.hpp>
namespace nb = nanobind;

// NB_MODULE(_core, m) {
//   m.def("tyler_covariances",
//         [](Eigen::Ref<const Eigen::MatrixXd> X, int k, double tol, int max_iter) {
//           nb::gil_scoped_release release;   // let Python threads run
//           return CovBIM::tyler_covariances(X, k, tol, max_iter);
//         },
//         nb::arg("X"), nb::arg("k"), nb::arg("tol") = 1e-6, nb::arg("max_iter") = 50);
// }

// NB_MODULE(_core, m) {
//   m.def("the_tester",
//         [](int x) {
//           nb::gil_scoped_release release;   // let Python threads run
//           return Test::test_function(x);
//         },
//         nb::arg("x"));
// }


// P   = geodsd.tyler_covariances(X, k=30, joint_location=True, tol=1e-6)   # (N, n, n)
// F   = geodsd.spd_features(P, metric="log_euclidean")                     # logm once per point
// G   = geodsd.build_graph(X, P, k=15, alpha=2.0, beta=3.0, lam=1.0)       # CSR arrays
// d, pred = geodsd.dijkstra(G, source=0)
// d1  = geodsd.recompute_path_lengths(G_beta1, pred, source=0)             # beta=1 length