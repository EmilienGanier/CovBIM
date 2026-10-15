#pragma once

#include <vector>

#include <Eigen/Dense>

#include <graph.hpp>
#include <knn.hpp>  // RowMatrixXd

namespace PATH {

struct ShortestPathResult {
    Eigen::VectorXd distances;  // N: shortest-path distance from the source, +inf if unreachable
    GRAPH::IndexVector predecessors;   // N: previous node on the shortest path, -1 for the source / unreachable
    GRAPH::IndexVector order;          // reachable nodes in the order they were settled (starts with the source)
    // the point of the order vector is that it allows to compute the shortest-path lengths from the source to all reachable nodes in a single pass, without having to traverse the predecessor tree recursively.
};

/// Dijkstra's algorithm from a single source. Edge weights must be non-negative.
/// Throws std::invalid_argument if the source is out of range.
ShortestPathResult dijkstra(const GRAPH::CsrGraph& graph, Eigen::Index source);

/// Node sequence source -> target following the predecessors; empty if target is unreachable.
std::vector<Eigen::Index> extract_path(const GRAPH::IndexVector& predecessors, Eigen::Index source,
                                       Eigen::Index target);

/// Distances from each source to every node: row s holds the distances from sources[s] (S x N).
/// Sources are processed in parallel.
KNN::RowMatrixXd distance_matrix(const GRAPH::CsrGraph& graph, const GRAPH::IndexVector& sources);

}