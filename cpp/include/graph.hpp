#pragma once

#include <Eigen/Dense>

namespace GRAPH {

using IndexVector = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

/// Weighted undirected graph in CSR format (each edge stored in both directions).
/// Compatible with scipy.sparse.csr_matrix((weights, indices, indptr), shape=(n_nodes, n_nodes)).
struct CsrGraph {
    Eigen::Index n_nodes = 0;
    IndexVector indptr;       // n_nodes + 1, indptr[i+1] - indptr[i] = degree of node i
    IndexVector indices;      // nnz, sorted within each row (nnz is the number of edges, counting each edge twice), indices[indptr[i]:indptr[i+1]] are the neighbors of node i
    Eigen::VectorXd weights;  // nnz, non-negative
};

}