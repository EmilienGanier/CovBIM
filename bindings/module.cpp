// bindings/module.cpp
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
// #include <geodsd/covariance.hpp>
#include <test.hpp>
namespace nb = nanobind;

// NB_MODULE(_core, m) {
//   m.def("tyler_covariances",
//         [](Eigen::Ref<const Eigen::MatrixXd> X, int k, double tol, int max_iter) {
//           nb::gil_scoped_release release;   // let Python threads run
//           return CovBIM::tyler_covariances(X, k, tol, max_iter);
//         },
//         nb::arg("X"), nb::arg("k"), nb::arg("tol") = 1e-6, nb::arg("max_iter") = 50);
// }

NB_MODULE(_core, m) {
  m.def("the_tester",
        [](int x) {
          nb::gil_scoped_release release;   // let Python threads run
          return Test::test_function(x);
        },
        nb::arg("x"));
}