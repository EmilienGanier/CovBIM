// bindings/module.cpp
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>
// #include <geodsd/covariance.hpp>
// #include <test.hpp>

#include <covariance.hpp>
#include <metric.hpp>
#include <shortest_path.hpp>

namespace nb = nanobind;

// NB_MODULE(_core, m) {
//   m.def("tyler_covariances",
//         [](Eigen::Ref<const Eigen::MatrixXd> X, int k, double tol, int max_iter) {
//           nb::gil_scoped_release release;   // let Python threads run
//           return CovBIM::tyler_covariances(X, k, tol, max_iter);
//         },
//         nb::arg("X"), nb::arg("k"), nb::arg("tol") = 1e-6, nb::arg("max_iter") = 50);
// }

using IndexVector = GRAPH::IndexVector;
using RowMatrixXd = KNN::RowMatrixXd;
using RowMatrixXi = KNN::RowMatrixXi;

using namespace TYLER;

NB_MODULE(_core, m) {

    // Graph of nearest neighbours in ambient space
    m.def("knn",
          [](Eigen::Ref<const RowMatrixXd> points, int k, bool include_self) {
              nb::gil_scoped_release release;
              KNN::KnnResult r = KNN::knn(points, k, include_self);
              return std::make_tuple(std::move(r.indices), std::move(r.distances));
          },
          nb::arg("points"), nb::arg("k"), nb::arg("include_self") = false);

    // list of kernels for the Tyler estimator
    nb::enum_<Kernel>(m, "Kernel")
        .value("Triweight", Kernel::Triweight)
        .value("Bump", Kernel::Bump)
        .value("Dumb", Kernel::Dumb);

    // Tyler M-estimator of scatter, computed at every point of `points` (N x n).
    m.def("tyler_covariances",
        [](Eigen::Ref<const RowMatrixXd> points, int k, Kernel kernel,
            bool joint_location, double tol, int max_iter, double ridge) {
            TylerParams p;
            p.k = k; p.kernel = kernel; p.joint_location = joint_location;
            p.tol = tol; p.max_iter = max_iter; p.ridge = ridge;

            TylerResult r;
            {
                nb::gil_scoped_release release;
                r = tyler_covariances(points, p);
            }
            return std::make_tuple(std::move(r.covariances), std::move(r.locations),
                                    std::move(r.radii), std::move(r.iterations),
                                    std::move(r.converged));
        },
        nb::arg("points"), nb::arg("k") = 30, nb::arg("kernel") = Kernel::Dumb,
        nb::arg("joint_location") = true, nb::arg("tol") = 1e-6, nb::arg("max_iter") = 500,
        nb::arg("ridge") = 1e-10);
    
    // list of SPD metrics for the edge weights (some may not be implemented yet)
    nb::enum_<METRIC::SpdMetric>(m, "SpdMetric")
        .value("Frobenius", METRIC::SpdMetric::Frobenius)
        .value("LogEuclidean", METRIC::SpdMetric::LogEuclidean)
        .value("Airm", METRIC::SpdMetric::Airm);

    // Define the class EdgeWeighter and its methods, including the constructor, nn graph, and weight function.
    nb::class_<METRIC::EdgeWeighter>(m, "EdgeWeighter")
        .def("__init__",
            [](METRIC::EdgeWeighter* self, Eigen::Ref<const RowMatrixXd> points,
                Eigen::Ref<const RowMatrixXd> covariances, double alpha, double beta,
                double lam, METRIC::SpdMetric spd_metric, double min_eig) {
                METRIC::MetricParams p;
                p.alpha = alpha; p.beta = beta; p.lambda = lam;
                p.spd_metric = spd_metric; p.min_eig = min_eig;
                new (self) METRIC::EdgeWeighter(points, covariances, p);
            },
            nb::arg("points"), nb::arg("covariances"), nb::arg("alpha") = 1.0,
            nb::arg("beta") = 1.0, nb::arg("lam") = 1.0,
            nb::arg("spd_metric") = METRIC::SpdMetric::LogEuclidean, nb::arg("min_eig") = 1e-12)
        .def("weight", &METRIC::EdgeWeighter::weight, nb::arg("i"), nb::arg("j"), nb::arg("beta"))
        .def("build_graph",
            [](const METRIC::EdgeWeighter& w, Eigen::Ref<const RowMatrixXi> neighbor_indices) {
                GRAPH::CsrGraph g;
                {
                    nb::gil_scoped_release release;
                    g = METRIC::build_graph(w, neighbor_indices);
                }
                return std::make_tuple(std::move(g.indptr), std::move(g.indices), std::move(g.weights));
            },
            nb::arg("neighbor_indices"))
        .def("tree_lengths",
            [](const METRIC::EdgeWeighter& w, const IndexVector& pred,
                const IndexVector& order, Eigen::Index source, double beta) {
                return METRIC::tree_lengths(w, pred, order, source, beta);
            },
            nb::arg("predecessors"), nb::arg("order"), nb::arg("source"), nb::arg("beta") = 1.0);

    // Dijkstra's algorithm for shortest paths in a weighted csr graph, returning distances, predecessors, and order of settlement.
    m.def("dijkstra",
        [](const IndexVector& indptr, const IndexVector& indices,
            const Eigen::VectorXd& weights, Eigen::Index source) {
            GRAPH::CsrGraph g;
            g.n_nodes = indptr.size() - 1;
            g.indptr = indptr; g.indices = indices; g.weights = weights;
            PATH::ShortestPathResult r;
            {
                nb::gil_scoped_release release;
                r = PATH::dijkstra(g, source);
            }
            return std::make_tuple(std::move(r.distances), std::move(r.predecessors), std::move(r.order));
        },
        nb::arg("indptr"), nb::arg("indices"), nb::arg("weights"), nb::arg("source"));
}


// P   = geodsd.tyler_covariances(X, k=30, joint_location=True, tol=1e-6)   # (N, n, n)
// F   = geodsd.spd_features(P, metric="log_euclidean")                     # logm once per point
// G   = geodsd.build_graph(X, P, k=15, alpha=2.0, beta=3.0, lam=1.0)       # CSR arrays
// d, pred = geodsd.dijkstra(G, source=0)
// d1  = geodsd.recompute_path_lengths(G_beta1, pred, source=0)             # beta=1 length