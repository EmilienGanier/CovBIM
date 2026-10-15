#include <shortest_path.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace PATH {

ShortestPathResult dijkstra(const GRAPH::CsrGraph& graph, Eigen::Index source) {
    const Eigen::Index N = graph.n_nodes;
    if (source < 0 || source >= N) {
        throw std::invalid_argument("dijkstra: source out of range");
    }

    ShortestPathResult res;
    res.distances = Eigen::VectorXd::Constant(N, std::numeric_limits<double>::infinity());
    res.predecessors = GRAPH::IndexVector::Constant(N, -1);
    res.order.resize(N);
    Eigen::Index n_settled = 0;

    using Item = std::pair<double, Eigen::Index>;  // (distance, node)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;

    res.distances(source) = 0.0;
    heap.emplace(0.0, source);

    while (!heap.empty()) {
        const auto [d, u] = heap.top();
        heap.pop();
        if (d > res.distances(u)) continue;  // outdated heap entry
        res.order(n_settled++) = u;

        for (Eigen::Index e = graph.indptr(u); e < graph.indptr(u + 1); ++e) {
            const Eigen::Index v = graph.indices(e);
            const double nd = d + graph.weights(e);
            if (nd < res.distances(v)) {
                res.distances(v) = nd;
                res.predecessors(v) = u;
                heap.emplace(nd, v);
            }
        }
    }
    res.order.conservativeResize(n_settled);
    return res;
}

std::vector<Eigen::Index> extract_path(const GRAPH::IndexVector& predecessors, Eigen::Index source,
                                       Eigen::Index target) {
    const Eigen::Index N = predecessors.size();
    if (source < 0 || source >= N || target < 0 || target >= N) {
        throw std::invalid_argument("extract_path: node index out of range");
    }

    std::vector<Eigen::Index> path;
    Eigen::Index v = target;
    while (v != source) {
        if (v < 0 || static_cast<Eigen::Index>(path.size()) > N) return {};  // unreachable / corrupted tree
        path.push_back(v);
        v = predecessors(v);
    }
    path.push_back(source);
    std::reverse(path.begin(), path.end());
    return path;
}

KNN::RowMatrixXd distance_matrix(const GRAPH::CsrGraph& graph, const GRAPH::IndexVector& sources) {
    const Eigen::Index S = sources.size();
    for (Eigen::Index s = 0; s < S; ++s) {  // validate first: no exception inside the parallel region
        if (sources(s) < 0 || sources(s) >= graph.n_nodes) {
            throw std::invalid_argument("distance_matrix: source out of range");
        }
    }

    KNN::RowMatrixXd out(S, graph.n_nodes);
#pragma omp parallel for schedule(dynamic)
    for (Eigen::Index s = 0; s < S; ++s) {
        out.row(s) = dijkstra(graph, sources(s)).distances.transpose();
    }
    return out;
}

}