#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <vector>

namespace setdetect::interpolation {

// Thin-plate-spline radial basis function interpolator, mirroring
// scipy.interpolate.RBFInterpolator(kernel="thin_plate_spline"): the interpolant is
// f(x) = sum_i w_i * TPS(||x - nodes_i||) + (a degree-1 polynomial in x). The
// polynomial term isn't optional decoration -- without it the interpolant has no way
// to represent an affine field (e.g. a pure translation) correctly once queried near or
// outside the convex hull of the training nodes, which is the common case for real
// tracking data. Solves the standard augmented system:
//   (K + smoothing*I) * w + P * c = outputs
//   P^T * w                       = 0
// where K is the RBF kernel matrix and P's rows are [1, x, y, ...] per node (the
// minimum polynomial degree scipy uses for this kernel). smoothing=0 (the default,
// matching scipy's own class default) gives exact interpolation through every node;
// smoothing>0 regularizes the fit instead of passing exactly through (noisy) points.
class ThinPlateSplineRBF {
  public:
    explicit ThinPlateSplineRBF(const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& outputs,
                                double smoothing = 0.0)
        : nodes_(inputs) {
        const auto n = static_cast<Eigen::Index>(inputs.size());
        const Eigen::Index in_dim = inputs[0].size();
        const auto out_dim = static_cast<Eigen::Index>(outputs[0].size());
        const Eigen::Index poly_dim = in_dim + 1;

        Eigen::MatrixXd kmat(n, n);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = 0; j < n; ++j) {
                const double r = (inputs[static_cast<size_t>(i)] - inputs[static_cast<size_t>(j)]).norm();
                kmat(i, j) = evaluateTPS(r);
            }
        }
        kmat.diagonal().array() += smoothing;

        Eigen::MatrixXd poly(n, poly_dim);
        for (Eigen::Index i = 0; i < n; ++i) {
            poly(i, 0) = 1.0;
            poly.row(i).tail(in_dim) = inputs[static_cast<size_t>(i)].transpose();
        }

        Eigen::MatrixXd lhs(n + poly_dim, n + poly_dim);
        lhs.setZero();
        lhs.topLeftCorner(n, n) = kmat;
        lhs.topRightCorner(n, poly_dim) = poly;
        lhs.bottomLeftCorner(poly_dim, n) = poly.transpose();

        Eigen::MatrixXd rhs(n + poly_dim, out_dim);
        rhs.setZero();
        for (Eigen::Index i = 0; i < n; ++i) {
            rhs.row(i) = outputs[static_cast<size_t>(i)];
        }

        // Solve for weights (RBF coefficients) and poly_coeffs_ (polynomial coefficients) together.
        const Eigen::MatrixXd coeffs = lhs.colPivHouseholderQr().solve(rhs);
        weights_ = coeffs.topRows(n);
        poly_coeffs_ = coeffs.bottomRows(poly_dim);
    }

    [[nodiscard]] auto interpolate(const Eigen::VectorXd& point) const -> Eigen::VectorXd {
        const auto out_dim = weights_.cols();
        Eigen::VectorXd result = Eigen::VectorXd::Zero(out_dim);

        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(nodes_.size()); ++i) {
            const double r = (point - nodes_[static_cast<size_t>(i)]).norm();
            result += weights_.row(i) * evaluateTPS(r);
        }
        result += poly_coeffs_.row(0);
        for (Eigen::Index k = 0; k < point.size(); ++k) {
            result += poly_coeffs_.row(k + 1) * point(k);
        }
        return result;
    }

  private:
    std::vector<Eigen::VectorXd> nodes_;
    Eigen::MatrixXd weights_;
    Eigen::MatrixXd poly_coeffs_;
    static auto evaluateTPS(double r) -> double {
        if (r <= 0.0) {
            return 0.0;
        }
        return r * r * std::log(r);
    }
};

} // namespace setdetect::interpolation
