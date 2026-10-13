#include <knn.hpp>

#include <nanoflann.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace KNN {
namespace {

// Minimal adaptor telling nanoflann how to read the point cloud (no copy of the data).
struct PointCloudAdaptor {
    Eigen::Ref<const RowMatrixXd> pts;

    explicit PointCloudAdaptor(const Eigen::Ref<const RowMatrixXd>& p) : pts(p) {}

    size_t kdtree_get_point_count() const { return static_cast<size_t>(pts.rows()); }
    double kdtree_get_pt(const size_t idx, const size_t dim) const { return pts(idx, dim); }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const {
        return false;  // let nanoflann compute the bounding box itself
    }
};

using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>,
    PointCloudAdaptor,
    -1,            // dimension known at runtime
    Eigen::Index>; // index type

}  // namespace

KnnResult knn(const Eigen::Ref<const RowMatrixXd>& points, int k, bool include_self) {
    const Eigen::Index N = points.rows();
    const Eigen::Index dim = points.cols();

    if (k < 1) {
        throw std::invalid_argument("knn: k must be >= 1");
    }
    // When excluding the point itself we query one extra neighbor and drop it.
    const Eigen::Index n_query = include_self ? k : k + 1;
    if (n_query > N) {
        throw std::invalid_argument("knn: not enough points for the requested k");
    }

    PointCloudAdaptor cloud(points);
    KdTree tree(static_cast<int>(dim), cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    // (nanoflann >= 1.5 builds the index in the constructor)

    KnnResult out;
    out.indices.resize(N, k);
    out.distances.resize(N, k);

#pragma omp parallel
    {
        std::vector<Eigen::Index> idx(n_query);  // per-thread buffers
        std::vector<double> dist2(n_query);

#pragma omp for schedule(static)
        for (Eigen::Index i = 0; i < N; ++i) {
            nanoflann::KNNResultSet<double, Eigen::Index> result(n_query);
            result.init(idx.data(), dist2.data());
            tree.findNeighbors(result, points.row(i).data());

            // Copy to the output, skipping the query point itself if requested.
            // (With duplicate points the point may not be first, so test the index.)
            Eigen::Index col = 0;
            for (Eigen::Index j = 0; j < n_query && col < k; ++j) {
                if (!include_self && idx[j] == i) continue;
                out.indices(i, col) = idx[j];
                out.distances(i, col) = std::sqrt(dist2[j]);  // nanoflann returns squared L2
                ++col;
            }
        }
    }
    return out;
}

}