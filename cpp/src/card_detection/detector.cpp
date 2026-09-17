#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <set>
#include <setdetect/card_detection/detector.hpp>
#include <setdetect/set_game/game.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace setdetect::card_detection {

namespace {

// Width:height of the flat card, matching model::card_classification::CARD_SIZE (224, 320).
constexpr double kCardAspectRatio = 224.0 / 320.0;
const std::array<Point2d, 4> kCanon = {Point2d{0.0, 0.0}, Point2d{kCardAspectRatio, 0.0},
                                       Point2d{kCardAspectRatio, 1.0}, Point2d{0.0, 1.0}};

auto quad_area(const std::array<Point2d, 4>& quad) -> double {
    double area = 0.0;
    for (int i = 0; i < 4; ++i) {
        const Point2d& a = quad[i];
        const Point2d& b = quad[(i + 1) % 4];
        area += a.x * b.y - b.x * a.y;
    }
    return std::abs(area) * 0.5;
}

auto apply_h(const Eigen::Matrix3d& mat, const std::array<Point2d, 4>& pts) -> std::array<Point2d, 4> {
    std::array<Point2d, 4> out{};
    for (int i = 0; i < 4; ++i) {
        const double x = pts[i].x;
        const double y = pts[i].y;
        const double denom = mat(2, 0) * x + mat(2, 1) * y + mat(2, 2);
        const double px = (mat(0, 0) * x + mat(0, 1) * y + mat(0, 2)) / denom;
        const double py = (mat(1, 0) * x + mat(1, 1) * y + mat(1, 2)) / denom;
        out[i] = {px, py};
    }
    return out;
}

// Orthogonal Procrustes fit of a rigid (rotation + translation) pose mapping the canonical
// rectangle onto the given (already plane-projected) quad.
auto procrustes_pose(const std::array<Point2d, 4>& world) -> std::array<double, 3> {
    Point2d world_mean{0.0, 0.0};
    Point2d canon_mean{0.0, 0.0};
    for (int i = 0; i < 4; ++i) {
        world_mean.x += world[i].x;
        world_mean.y += world[i].y;
        canon_mean.x += kCanon[i].x;
        canon_mean.y += kCanon[i].y;
    }
    world_mean.x /= 4.0;
    world_mean.y /= 4.0;
    canon_mean.x /= 4.0;
    canon_mean.y /= 4.0;

    Eigen::Matrix2d m = Eigen::Matrix2d::Zero();
    for (int i = 0; i < 4; ++i) {
        const Eigen::Vector2d canon_c(kCanon[i].x - canon_mean.x, kCanon[i].y - canon_mean.y);
        const Eigen::Vector2d world_c(world[i].x - world_mean.x, world[i].y - world_mean.y);
        m += canon_c * world_c.transpose();
    }

    const Eigen::JacobiSVD<Eigen::Matrix2d> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix2d u = svd.matrixU();
    Eigen::Matrix2d v = svd.matrixV();
    Eigen::Matrix2d rot = v * u.transpose();
    if (rot.determinant() < 0.0) {
        v.col(1) *= -1.0;
        rot = v * u.transpose();
    }

    const Eigen::Vector2d canon_mean_vec(canon_mean.x, canon_mean.y);
    const Eigen::Vector2d world_mean_vec(world_mean.x, world_mean.y);
    const Eigen::Vector2d t = world_mean_vec - rot * canon_mean_vec;

    return {std::atan2(rot(1, 0), rot(0, 0)), t.x(), t.y()};
}

// Exact 4-point perspective transform (mirrors cv::getPerspectiveTransform): solves the
// determined 8x8 linear system for the homography mapping src[i] -> dst[i] exactly.
auto solve_perspective_transform(const std::array<Point2d, 4>& src, const std::array<Point2d, 4>& dst)
    -> std::optional<Eigen::Matrix3d> {
    Eigen::Matrix<double, 8, 8> m = Eigen::Matrix<double, 8, 8>::Zero();
    Eigen::Matrix<double, 8, 1> b;
    for (Eigen::Index i = 0; i < 4; ++i) {
        const double x = src[static_cast<size_t>(i)].x;
        const double y = src[static_cast<size_t>(i)].y;
        const double u = dst[static_cast<size_t>(i)].x;
        const double v = dst[static_cast<size_t>(i)].y;
        m(2 * i, 0) = x;
        m(2 * i, 1) = y;
        m(2 * i, 2) = 1.0;
        m(2 * i, 6) = -x * u;
        m(2 * i, 7) = -y * u;
        b(2 * i) = u;
        m((2 * i) + 1, 3) = x;
        m((2 * i) + 1, 4) = y;
        m((2 * i) + 1, 5) = 1.0;
        m((2 * i) + 1, 6) = -x * v;
        m((2 * i) + 1, 7) = -y * v;
        b((2 * i) + 1) = v;
    }

    const Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 8, 8>> qr(m);
    if (qr.rank() < 8) {
        return std::nullopt;
    }
    const Eigen::Matrix<double, 8, 1> h = qr.solve(b);
    if (!h.allFinite()) {
        return std::nullopt;
    }
    Eigen::Matrix3d out;
    // clang-format off
    out << h(0), h(1), h(2),
           h(3), h(4), h(5),
           h(6), h(7), 1.0;
    // clang-format on
    return out;
}

// Quad -> canonical homography, with a rectified-box fallback for degenerate quads.
auto anchor_homography(const std::array<Point2d, 4>& quad) -> Eigen::Matrix3d {
    auto usable = [](const std::optional<Eigen::Matrix3d>& h) {
        return h.has_value() && h->allFinite() && std::abs(h->determinant()) > 1e-9;
    };

    std::optional<Eigen::Matrix3d> img2world = solve_perspective_transform(quad, kCanon);
    if (!usable(img2world)) {
        double x_min = quad[0].x;
        double x_max = quad[0].x;
        double y_min = quad[0].y;
        double y_max = quad[0].y;
        for (const Point2d& p : quad) {
            x_min = std::min(x_min, p.x);
            x_max = std::max(x_max, p.x);
            y_min = std::min(y_min, p.y);
            y_max = std::max(y_max, p.y);
        }
        constexpr double kPad = 4.0;
        const std::array<Point2d, 4> box = {Point2d{x_min - kPad, y_min - kPad}, Point2d{x_max + kPad, y_min - kPad},
                                            Point2d{x_max + kPad, y_max + kPad}, Point2d{x_min - kPad, y_max + kPad}};
        img2world = solve_perspective_transform(box, kCanon);
    }
    if (!usable(img2world)) {
        throw std::runtime_error("could not build an invertible anchor homography");
    }
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access) usable() above already checked has_value()
    return *img2world;
}

// Residual and analytic Jacobian of the reprojection over all corners, for the joint fit of
// the shared plane homography (params 0..7, world2img with h(2,2) fixed to 1) and the
// per-card rigid poses (3 params per free card: theta, tx, ty). The anchor card's pose is
// fixed at the canonical identity pose and is not part of `p`.
auto residual_and_jacobian(const Eigen::VectorXd& p, const std::vector<Point2d>& obs, int anchor,
                           const std::vector<int>& free, int n) -> std::pair<Eigen::VectorXd, Eigen::MatrixXd> {
    const auto n_params = static_cast<int>(p.size());
    const int n_pts = n * 4;

    const double h00 = p(0);
    const double h01 = p(1);
    const double h02 = p(2);
    const double h10 = p(3);
    const double h11 = p(4);
    const double h12 = p(5);
    const double h20 = p(6);
    const double h21 = p(7);

    std::vector<int> free_index_of(static_cast<size_t>(n), -1);
    for (size_t j = 0; j < free.size(); ++j) {
        free_index_of[static_cast<size_t>(free[j])] = static_cast<int>(j);
    }

    std::vector<std::array<Point2d, 4>> world(static_cast<size_t>(n));
    std::vector<double> cos_t(static_cast<size_t>(n), 1.0);
    std::vector<double> sin_t(static_cast<size_t>(n), 0.0);
    world[static_cast<size_t>(anchor)] = kCanon;
    for (size_t j = 0; j < free.size(); ++j) {
        const int i = free[j];
        const int base = 8 + (3 * static_cast<int>(j));
        const double theta = p(base);
        const double tx = p(base + 1);
        const double ty = p(base + 2);
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        cos_t[static_cast<size_t>(i)] = c;
        sin_t[static_cast<size_t>(i)] = s;
        for (int k = 0; k < 4; ++k) {
            const Point2d& cc = kCanon[static_cast<size_t>(k)];
            world[static_cast<size_t>(i)][static_cast<size_t>(k)] = {(c * cc.x) - (s * cc.y) + tx,
                                                                     (s * cc.x) + (c * cc.y) + ty};
        }
    }

    const auto n_residuals = static_cast<Eigen::Index>(n_pts) * 2;
    Eigen::VectorXd residual(n_residuals);
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(n_residuals, n_params);

    for (int i = 0; i < n; ++i) {
        const int free_j = free_index_of[static_cast<size_t>(i)];
        for (int k = 0; k < 4; ++k) {
            const int pt_idx = (i * 4) + k;
            const double x = world[static_cast<size_t>(i)][static_cast<size_t>(k)].x;
            const double y = world[static_cast<size_t>(i)][static_cast<size_t>(k)].y;
            const double denom = (h20 * x) + (h21 * y) + 1.0;
            const double m = (h00 * x) + (h01 * y) + h02;
            const double ns = (h10 * x) + (h11 * y) + h12;
            const double pred_x = m / denom;
            const double pred_y = ns / denom;

            const int row_x = 2 * pt_idx;
            const int row_y = row_x + 1;
            residual(row_x) = pred_x - obs[static_cast<size_t>(pt_idx)].x;
            residual(row_y) = pred_y - obs[static_cast<size_t>(pt_idx)].y;

            const double denom2 = denom * denom;
            jacobian(row_x, 0) = x / denom;
            jacobian(row_x, 1) = y / denom;
            jacobian(row_x, 2) = 1.0 / denom;
            jacobian(row_y, 3) = x / denom;
            jacobian(row_y, 4) = y / denom;
            jacobian(row_y, 5) = 1.0 / denom;
            const double dpx_dx = ((h00 * denom) - (m * h20)) / denom2;
            const double dpx_dy = ((h01 * denom) - (m * h21)) / denom2;
            const double dpy_dx = ((h10 * denom) - (ns * h20)) / denom2;
            const double dpy_dy = ((h11 * denom) - (ns * h21)) / denom2;
            jacobian(row_x, 6) = -m * x / denom2;
            jacobian(row_x, 7) = -m * y / denom2;
            jacobian(row_y, 6) = -ns * x / denom2;
            jacobian(row_y, 7) = -ns * y / denom2;

            if (free_j >= 0) {
                const int base = 8 + (3 * free_j);
                const double c = cos_t[static_cast<size_t>(i)];
                const double s = sin_t[static_cast<size_t>(i)];
                const Point2d& cc = kCanon[static_cast<size_t>(k)];
                const double dwx_dtheta = (-s * cc.x) - (c * cc.y);
                const double dwy_dtheta = (c * cc.x) - (s * cc.y);
                jacobian(row_x, base) = (dpx_dx * dwx_dtheta) + (dpx_dy * dwy_dtheta);
                jacobian(row_x, base + 1) = dpx_dx;
                jacobian(row_x, base + 2) = dpx_dy;
                jacobian(row_y, base) = (dpy_dx * dwx_dtheta) + (dpy_dy * dwy_dtheta);
                jacobian(row_y, base + 1) = dpy_dx;
                jacobian(row_y, base + 2) = dpy_dy;
            }
        }
    }

    return {residual, jacobian};
}

struct LMResult {
    Eigen::VectorXd x;
    double cost = 0.0;
};

// Small dense Levenberg-Marquardt least-squares solver over an analytic residual+Jacobian.
template <typename ResidualJacFn>
auto levenberg_marquardt(const Eigen::VectorXd& p0, ResidualJacFn&& residual_jac_fn, int max_iterations = 50,
                         double lambda_init = 1e-3) -> LMResult {
    constexpr double kEps = 1e-10;
    Eigen::VectorXd p = p0;
    auto [res, jac] = residual_jac_fn(p);
    double cost = 0.5 * res.squaredNorm();
    double lambda = lambda_init;

    for (int iter = 0; iter < max_iterations; ++iter) {
        const Eigen::MatrixXd jtj = jac.transpose() * jac;
        const Eigen::VectorXd jtr = jac.transpose() * res;
        Eigen::MatrixXd damped = jtj;
        for (Eigen::Index i = 0; i < damped.rows(); ++i) {
            damped(i, i) += lambda * std::max(jtj(i, i), kEps);
        }
        const Eigen::VectorXd delta = damped.ldlt().solve(-jtr);
        if (!delta.allFinite()) {
            break;
        }
        const Eigen::VectorXd p_new = p + delta;
        auto [res_new, jac_new] = residual_jac_fn(p_new);
        const double cost_new = 0.5 * res_new.squaredNorm();

        if (cost_new < cost) {
            const double improvement = cost - cost_new;
            const bool converged = improvement < kEps * (1.0 + cost) || delta.norm() < kEps * (1.0 + p_new.norm());
            p = p_new;
            res = std::move(res_new);
            jac = std::move(jac_new);
            cost = cost_new;
            lambda = std::max(lambda * 0.3, 1e-12);
            if (converged) {
                break;
            }
        } else {
            lambda *= 10.0;
            if (lambda > 1e12) {
                break;
            }
        }
    }
    return {p, cost};
}

// Jointly fits the shared plane homography (world->image) and per-card rigid poses.
//
// The arrangement space is the world plane of the cards, where every card is a rectangle
// of known aspect ratio. Returns the world->image homography.
auto fit_arrangement(const std::vector<std::array<Point2d, 4>>& corners) -> Eigen::Matrix3d {
    const auto n = static_cast<int>(corners.size());

    std::vector<Point2d> obs;
    obs.reserve(static_cast<size_t>(n) * 4);
    for (const auto& c : corners) {
        obs.insert(obs.end(), c.begin(), c.end());
    }

    int anchor = 0;
    double best_area = -1.0;
    for (int i = 0; i < n; ++i) {
        const double area = quad_area(corners[static_cast<size_t>(i)]);
        if (area > best_area) {
            best_area = area;
            anchor = i;
        }
    }

    const Eigen::Matrix3d img2world = anchor_homography(corners[static_cast<size_t>(anchor)]);
    Eigen::Matrix3d world2img = img2world.inverse();
    world2img /= world2img(2, 2);

    std::vector<int> free;
    free.reserve(static_cast<size_t>(n - 1));
    for (int i = 0; i < n; ++i) {
        if (i != anchor) {
            free.push_back(i);
        }
    }

    std::vector<std::array<double, 3>> poses(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        poses[static_cast<size_t>(i)] = procrustes_pose(apply_h(img2world, corners[static_cast<size_t>(i)]));
    }
    poses[static_cast<size_t>(anchor)] = {0.0, 0.0, 0.0};

    const int n_params = 8 + (3 * static_cast<int>(free.size()));
    Eigen::VectorXd p0(n_params);
    p0(0) = world2img(0, 0);
    p0(1) = world2img(0, 1);
    p0(2) = world2img(0, 2);
    p0(3) = world2img(1, 0);
    p0(4) = world2img(1, 1);
    p0(5) = world2img(1, 2);
    p0(6) = world2img(2, 0);
    p0(7) = world2img(2, 1);
    for (size_t j = 0; j < free.size(); ++j) {
        const std::array<double, 3>& ps = poses[static_cast<size_t>(free[j])];
        const int base = 8 + (3 * static_cast<int>(j));
        p0(base) = ps[0];
        p0(base + 1) = ps[1];
        p0(base + 2) = ps[2];
    }

    const auto residual_jac = [&](const Eigen::VectorXd& p) { return residual_and_jacobian(p, obs, anchor, free, n); };

    const auto [res0, jac0] = residual_jac(p0);
    const double cost0 = 0.5 * res0.squaredNorm();
    const LMResult result = levenberg_marquardt(p0, residual_jac);
    const Eigen::VectorXd& p_final = (result.cost > cost0) ? p0 : result.x;

    Eigen::Matrix3d out;
    // clang-format off
    out << p_final(0), p_final(1), p_final(2),
           p_final(3), p_final(4), p_final(5),
           p_final(6), p_final(7), 1.0;
    // clang-format on
    return out;
}

// Even-odd (crossing number) point-in-polygon test; boundary points are not guaranteed to
// register as inside, mirroring the tolerance of the occlusion heuristic below.
auto point_in_quad(const std::array<Point2d, 4>& poly, const Point2d& p) -> bool {
    bool inside = false;
    for (int i = 0, j = 3; i < 4; j = i++) {
        const Point2d& pi = poly[static_cast<size_t>(i)];
        const Point2d& pj = poly[static_cast<size_t>(j)];
        const bool straddles = (pi.y > p.y) != (pj.y > p.y);
        if (straddles && (p.x < (((pj.x - pi.x) * (p.y - pi.y)) / (pj.y - pi.y)) + pi.x)) {
            inside = !inside;
        }
    }
    return inside;
}

// Number of card a's hidden corners that lie inside card b's quad.
auto hidden_in(const DetectedCard& a, const DetectedCard& b) -> int {
    int count = 0;
    for (int k = 0; k < 4; ++k) {
        if (a.corner_visibility[static_cast<size_t>(k)] >= kCornerVisibleThreshold) {
            continue;
        }
        if (point_in_quad(b.corners, a.corners[static_cast<size_t>(k)])) {
            ++count;
        }
    }
    return count;
}

struct OcclusionPartialOrder {
    std::vector<std::set<int>> down; // down[j] = cards that j covers
    std::vector<int> covers;         // covers[i] = number of cards covering card i
    std::vector<int> depth;          // depth[i] = hidden-corner count for card i
};

// Builds the pairwise "covers" evidence between every pair of cards, plus each card's
// occlusion depth (used to break ties/cycles deterministically).
auto build_occlusion_partial_order(const std::vector<DetectedCard>& cards) -> OcclusionPartialOrder {
    const auto n = static_cast<int>(cards.size());

    std::vector<std::set<int>> above(static_cast<size_t>(n)); // above[i] = cards covering card i
    for (int a = 0; a < n; ++a) {
        for (int b = a + 1; b < n; ++b) {
            const int a_in_b = hidden_in(cards[static_cast<size_t>(a)], cards[static_cast<size_t>(b)]);
            const int b_in_a = hidden_in(cards[static_cast<size_t>(b)], cards[static_cast<size_t>(a)]);
            if (a_in_b == b_in_a) {
                continue;
            }
            if (a_in_b > b_in_a) {
                above[static_cast<size_t>(a)].insert(b);
            } else {
                above[static_cast<size_t>(b)].insert(a);
            }
        }
    }

    std::vector<int> depth(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto& vis = cards[static_cast<size_t>(i)].corner_visibility;
        depth[static_cast<size_t>(i)] = static_cast<int>(
            std::count_if(vis.begin(), vis.end(), [](double v) { return v < kCornerVisibleThreshold; }));
    }

    std::vector<std::set<int>> down(static_cast<size_t>(n));
    std::vector<int> covers(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        covers[static_cast<size_t>(i)] = static_cast<int>(above[static_cast<size_t>(i)].size());
        for (const int j : above[static_cast<size_t>(i)]) {
            down[static_cast<size_t>(j)].insert(i);
        }
    }

    return {std::move(down), std::move(covers), std::move(depth)};
}

// Returns card indices from topmost to deepest, inferred from per-corner occlusion.
//
// A corner of card a hidden below card b (contained in b's quad with low corner
// visibility) is evidence that b is above a. The partial order from such pairwise evidence
// is topologically sorted; ambiguous/cyclic cases resolve deterministically by occlusion
// depth (hidden-corner count).
auto occlusion_order(const std::vector<DetectedCard>& cards) -> std::vector<int> {
    const auto n = static_cast<int>(cards.size());
    std::vector<int> order;
    order.reserve(static_cast<size_t>(n));
    if (n < 2) {
        for (int i = 0; i < n; ++i) {
            order.push_back(i);
        }
        return order;
    }

    auto [down, covers, depth] = build_occlusion_partial_order(cards);

    auto by_depth = [&](int a, int b) {
        if (depth[static_cast<size_t>(a)] != depth[static_cast<size_t>(b)]) {
            return depth[static_cast<size_t>(a)] < depth[static_cast<size_t>(b)];
        }
        return a < b;
    };

    std::set<int> remaining;
    std::set<int> frontier;
    for (int i = 0; i < n; ++i) {
        remaining.insert(i);
        if (covers[static_cast<size_t>(i)] == 0) {
            frontier.insert(i);
        }
    }

    while (!remaining.empty()) {
        const int pick = frontier.empty() ? *std::min_element(remaining.begin(), remaining.end(), by_depth)
                                          : *std::min_element(frontier.begin(), frontier.end(), by_depth);
        order.push_back(pick);
        remaining.erase(pick);
        frontier.erase(pick);
        for (const int j : down[static_cast<size_t>(pick)]) {
            if (remaining.contains(j)) {
                covers[static_cast<size_t>(j)] -= 1;
                if (covers[static_cast<size_t>(j)] == 0) {
                    frontier.insert(j);
                }
            }
        }
    }
    return order;
}

// Width/height of the classifier's input card crop; must match the exported
// classification model's expected input size.
constexpr int kClassCardWidth = 224;
constexpr int kClassCardHeight = 320;
constexpr uint8_t kLetterboxPadValue = 114;

// Thin wrapper around cv::resize(INTER_LINEAR). Its pixel-center-aligned sampling
// convention (`sx = (x+0.5)*scale - 0.5`) must be preserved exactly: the model is
// trained on cv2-resized images, and a corner-aligned convention would silently
// bias inference.
auto resize_linear(const cv::Mat& img, int new_rows, int new_cols) -> cv::Mat {
    if (new_rows == img.rows && new_cols == img.cols) {
        return img.clone();
    }
    cv::Mat out;
    cv::resize(img, out, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
    return out;
}

// `m` is the forward (src -> dst) homography; cv::warpPerspective inverts it
// internally and samples img at each dst pixel's mapped source location,
// with out-of-bounds samples falling back to black (BORDER_CONSTANT,
// default zero scalar).
auto warp_perspective(const cv::Mat& img, const Eigen::Matrix3d& m, int out_rows, int out_cols) -> cv::Mat {
    cv::Mat m_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            m_cv.at<double>(r, c) = m(r, c);
        }
    }
    cv::Mat out;
    cv::warpPerspective(img, out, m_cv, cv::Size(out_cols, out_rows), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return out;
}

struct LetterboxResult {
    cv::Mat image;
    double ratio = 1.0;
    double pad_left = 0.0;
    double pad_top = 0.0;
};

// YOLO-style letterbox: aspect-preserving resize to fit inside target_h x target_w,
// then pad to exactly that size with a constant value, asymmetric rounding matching
// the ultralytics reference implementation.
auto letterbox(const cv::Mat& img, int target_h, int target_w) -> LetterboxResult {
    const int h = img.rows;
    const int w = img.cols;
    const double r = std::min(static_cast<double>(target_h) / h, static_cast<double>(target_w) / w);
    const int new_w = static_cast<int>(std::lround(w * r));
    const int new_h = static_cast<int>(std::lround(h * r));
    const cv::Mat resized = resize_linear(img, new_h, new_w);

    const double dw = (target_w - new_w) / 2.0;
    const double dh = (target_h - new_h) / 2.0;
    const int top = static_cast<int>(std::lround(dh - 0.1));
    const int left = static_cast<int>(std::lround(dw - 0.1));
    const int bottom = target_h - new_h - top;
    const int right = target_w - new_w - left;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar::all(kLetterboxPadValue));

    return {padded, r, static_cast<double>(left), static_cast<double>(top)};
}

// Greedy IoU-based non-maximum suppression; returns indices to keep, highest score
// first.
auto nms(const std::vector<std::array<double, 4>>& boxes_xyxy, const std::vector<double>& scores, double iou_threshold)
    -> std::vector<int> {
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)]; });

    std::vector<int> keep;
    std::vector<bool> suppressed(scores.size(), false);
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const int i = order[oi];
        if (suppressed[static_cast<size_t>(i)]) {
            continue;
        }
        keep.push_back(i);
        const std::array<double, 4>& bi = boxes_xyxy[static_cast<size_t>(i)];
        const double area_i = (bi[2] - bi[0]) * (bi[3] - bi[1]);
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            const int j = order[oj];
            if (suppressed[static_cast<size_t>(j)]) {
                continue;
            }
            const std::array<double, 4>& bj = boxes_xyxy[static_cast<size_t>(j)];
            const double xx1 = std::max(bi[0], bj[0]);
            const double yy1 = std::max(bi[1], bj[1]);
            const double xx2 = std::min(bi[2], bj[2]);
            const double yy2 = std::min(bi[3], bj[3]);
            const double inter = std::max(0.0, xx2 - xx1) * std::max(0.0, yy2 - yy1);
            const double area_j = (bj[2] - bj[0]) * (bj[3] - bj[1]);
            const double iou = inter / (area_i + area_j - inter + 1e-6);
            if (iou > iou_threshold) {
                suppressed[static_cast<size_t>(j)] = true;
            }
        }
    }
    return keep;
}

// Parses raw corner-keypoint model output already transposed to (n_rows, 17): columns
// [0:4]=box xywh, [4]=score, [5:17]=4 corners x (x, y, visibility) in letterboxed
// coordinates. Applies confidence filtering, NMS, and undoes the letterbox transform.
auto parse_corner_output(const float* rows, int n_rows, int n_cols, double conf_threshold, double iou_threshold,
                         int img_h, int img_w, double ratio, double pad_left, double pad_top)
    -> std::vector<CornerDetection> {
    std::vector<std::array<double, 4>> boxes_xyxy;
    std::vector<double> scores;
    std::vector<int> row_indices;
    for (int i = 0; i < n_rows; ++i) {
        const float* row = rows + (static_cast<size_t>(i) * static_cast<size_t>(n_cols));
        const double score = row[4];
        if (score <= conf_threshold) {
            continue;
        }
        const double cx = row[0];
        const double cy = row[1];
        const double bw = row[2];
        const double bh = row[3];
        boxes_xyxy.push_back({cx - (bw / 2.0), cy - (bh / 2.0), cx + (bw / 2.0), cy + (bh / 2.0)});
        scores.push_back(score);
        row_indices.push_back(i);
    }
    if (boxes_xyxy.empty()) {
        return {};
    }

    const std::vector<int> keep = nms(boxes_xyxy, scores, iou_threshold);

    std::vector<CornerDetection> results;
    results.reserve(keep.size());
    for (const int idx : keep) {
        const int orig_row = row_indices[static_cast<size_t>(idx)];
        const float* row = rows + (static_cast<size_t>(orig_row) * static_cast<size_t>(n_cols));
        CornerDetection det;
        for (int k = 0; k < 4; ++k) {
            const double kx = row[5 + (k * 3) + 0];
            const double ky = row[5 + (k * 3) + 1];
            const double kv = row[5 + (k * 3) + 2];
            double x = (kx - pad_left) / ratio;
            double y = (ky - pad_top) / ratio;
            x = std::clamp(x, 0.0, static_cast<double>(img_w - 1));
            y = std::clamp(y, 0.0, static_cast<double>(img_h - 1));
            det.corners[static_cast<size_t>(k)] = {x, y};
            det.visibility[static_cast<size_t>(k)] = kv;
        }
        results.push_back(det);
    }
    return results;
}

// Perspective-warps `quad` (in img_rgb's coordinates) to an out_w x out_h rectangle,
// then crops crop_frac off each edge and resizes back to out_w x out_h. Warps
// directly from the full image against the true quad coordinates rather than
// cropping to the quad's bounding box first: numerically equivalent, since only
// the destination raster is ever iterated. Returns nullopt for a degenerate
// (zero-area/non-invertible) quad.
auto warp_card(const cv::Mat& img_rgb, const std::array<Point2d, 4>& quad, int out_w, int out_h,
               double crop_frac = 0.06) -> std::optional<cv::Mat> {
    const std::array<Point2d, 4> dst = {Point2d{0.0, 0.0}, Point2d{static_cast<double>(out_w), 0.0},
                                        Point2d{static_cast<double>(out_w), static_cast<double>(out_h)},
                                        Point2d{0.0, static_cast<double>(out_h)}};
    const std::optional<Eigen::Matrix3d> h = solve_perspective_transform(quad, dst);
    if (!h.has_value() || !h->allFinite() || std::abs(h->determinant()) < 1e-9) {
        return std::nullopt;
    }
    cv::Mat warped = warp_perspective(img_rgb, *h, out_h, out_w);

    const int cw = static_cast<int>(std::lround(crop_frac * out_w));
    const int ch = static_cast<int>(std::lround(crop_frac * out_h));
    if ((2 * cw) >= out_w || (2 * ch) >= out_h) {
        return warped;
    }
    const int crop_h = out_h - (2 * ch);
    const int crop_w = out_w - (2 * cw);
    const cv::Mat cropped = warped(cv::Rect(cw, ch, crop_w, crop_h));
    return resize_linear(cropped, out_h, out_w);
}

auto softmax4(const std::array<double, 4>& logits) -> std::array<double, 4> {
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    std::array<double, 4> exp_vals{};
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        exp_vals[static_cast<size_t>(i)] = std::exp(logits[static_cast<size_t>(i)] - max_logit);
        sum += exp_vals[static_cast<size_t>(i)];
    }
    std::array<double, 4> out{};
    for (int i = 0; i < 4; ++i) {
        out[static_cast<size_t>(i)] = exp_vals[static_cast<size_t>(i)] / sum;
    }
    return out;
}

auto argmax4(const std::array<double, 4>& probs) -> int {
    return static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

// Fetches every output name of `session`, keeping the backing allocations alive in
// `holders` for as long as the returned pointers are used.
auto all_output_names(Ort::Session& session, std::vector<Ort::AllocatedStringPtr>& holders)
    -> std::vector<const char*> {
    const Ort::AllocatorWithDefaultOptions allocator;
    const size_t n = session.GetOutputCount();
    holders.reserve(n);
    std::vector<const char*> names;
    names.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        holders.push_back(session.GetOutputNameAllocated(i, allocator));
        names.push_back(holders.back().get());
    }
    return names;
}

// Index of `target` within `names`, or throws if the ONNX graph doesn't declare it.
auto find_output_index(const std::vector<const char*>& names, const std::string& target) -> size_t {
    for (size_t i = 0; i < names.size(); ++i) {
        if (target == names[i]) {
            return i;
        }
    }
    throw std::runtime_error("ONNX classifier output not found: " + target);
}

constexpr std::array<double, 3> kClassMean = {0.485, 0.456, 0.406};
constexpr std::array<double, 3> kClassStd = {0.229, 0.224, 0.225};

// Appends one card_rgb image's CHW-normalized ((x/255 - MEAN) / STD) float32 data to
// the flat NCHW `batch` buffer.
void append_normalized_card(const cv::Mat& card_rgb, std::vector<float>& batch) {
    const int h = card_rgb.rows;
    const int w = card_rgb.cols;
    const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w);
    const size_t base = batch.size();
    batch.resize(base + (plane * 3));
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = card_rgb.ptr(y);
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < 3; ++ch) {
                const double v =
                    ((static_cast<double>(row[(static_cast<size_t>(x) * 3) + static_cast<size_t>(ch)]) / 255.0) -
                     kClassMean[static_cast<size_t>(ch)]) /
                    kClassStd[static_cast<size_t>(ch)];
                batch[base + (static_cast<size_t>(ch) * plane) + (static_cast<size_t>(y) * static_cast<size_t>(w)) +
                      static_cast<size_t>(x)] = static_cast<float>(v);
            }
        }
    }
}

void warm_up_corner_session(Ort::Session& session, int imgsz) {
    std::vector<float> dummy(static_cast<size_t>(imgsz) * static_cast<size_t>(imgsz) * 3, 0.0F);
    const std::array<int64_t, 4> shape = {1, 3, imgsz, imgsz};
    const Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const Ort::Value input =
        Ort::Value::CreateTensor<float>(mem_info, dummy.data(), dummy.size(), shape.data(), shape.size());
    const Ort::AllocatorWithDefaultOptions allocator;
    const Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(0, allocator);
    const char* out_name_c = out_name.get();
    constexpr const char* kInputName = "images";
    session.Run(Ort::RunOptions{}, &kInputName, &input, 1, &out_name_c, 1);
}

void warm_up_class_session(Ort::Session& session) {
    std::vector<float> dummy(static_cast<size_t>(kClassCardWidth) * static_cast<size_t>(kClassCardHeight) * 3, 0.0F);
    const std::array<int64_t, 4> shape = {1, 3, kClassCardHeight, kClassCardWidth};
    const Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const Ort::Value input =
        Ort::Value::CreateTensor<float>(mem_info, dummy.data(), dummy.size(), shape.data(), shape.size());
    std::vector<Ort::AllocatedStringPtr> holders;
    const std::vector<const char*> names = all_output_names(session, holders);
    constexpr const char* kInputName = "input";
    session.Run(Ort::RunOptions{}, &kInputName, &input, 1, names.data(), names.size());
}

} // namespace

// A set_game::Card only when every head's argmax is a real class (index != 3/"other").
auto match_card(const DetectedCard& dc) -> std::optional<set_game::Card> {
    const int count_idx = argmax4(dc.count_probs);
    const int color_idx = argmax4(dc.color_probs);
    const int shape_idx = argmax4(dc.shape_probs);
    const int fill_idx = argmax4(dc.fill_probs);
    if (count_idx == 3 || color_idx == 3 || shape_idx == 3 || fill_idx == 3) {
        return std::nullopt;
    }
    return set_game::Card{static_cast<set_game::Count>(count_idx), static_cast<set_game::Color>(color_idx),
                          static_cast<set_game::Shape>(shape_idx), static_cast<set_game::Fill>(fill_idx)};
}

auto cards_arrangement(const std::vector<DetectedCard>& detected_cards, double inset) -> CardsArrangement {
    const auto n = static_cast<int>(detected_cards.size());
    if (n == 0) {
        return CardsArrangement{};
    }

    std::vector<std::array<Point2d, 4>> corners(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        corners[static_cast<size_t>(i)] = detected_cards[static_cast<size_t>(i)].corners;
    }

    const Eigen::Matrix3d world2img = fit_arrangement(corners);
    const Eigen::Matrix3d img2world = world2img.inverse();

    std::vector<std::array<Point2d, 4>> world_corners(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        world_corners[static_cast<size_t>(i)] = apply_h(img2world, corners[static_cast<size_t>(i)]);
    }

    const double align = std::atan2(world2img(1, 0), world2img(0, 0));
    const double cos_a = std::cos(align);
    const double sin_a = std::sin(align);
    for (auto& quad : world_corners) {
        for (Point2d& p : quad) {
            p = {(cos_a * p.x) - (sin_a * p.y), (sin_a * p.x) + (cos_a * p.y)};
        }
    }

    std::vector<double> rotations(static_cast<size_t>(n));
    std::vector<double> widths(static_cast<size_t>(n));
    std::vector<double> heights(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto& quad = world_corners[static_cast<size_t>(i)];
        const double dtop_x = quad[1].x - quad[0].x;
        const double dtop_y = quad[1].y - quad[0].y;
        rotations[static_cast<size_t>(i)] = std::atan2(dtop_y, dtop_x);
        widths[static_cast<size_t>(i)] = std::hypot(dtop_x, dtop_y);
        heights[static_cast<size_t>(i)] = std::hypot(quad[2].x - quad[1].x, quad[2].y - quad[1].y);
    }

    const double avg_w = std::accumulate(widths.begin(), widths.end(), 0.0) / n;
    const double avg_h = std::accumulate(heights.begin(), heights.end(), 0.0) / n;
    const double common_h = std::sqrt(avg_w * avg_h / kCardAspectRatio);
    const double common_w = common_h * kCardAspectRatio;

    std::vector<Point2d> centers(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        Point2d sum{0.0, 0.0};
        for (const Point2d& p : world_corners[static_cast<size_t>(i)]) {
            sum.x += p.x;
            sum.y += p.y;
        }
        centers[static_cast<size_t>(i)] = {sum.x / 4.0, sum.y / 4.0};
    }

    const double half_x = common_w / 2.0;
    const double half_y = common_h / 2.0;
    const std::array<Point2d, 4> half_offsets = {Point2d{half_x, half_y}, Point2d{half_x, -half_y},
                                                 Point2d{-half_x, -half_y}, Point2d{-half_x, half_y}};

    std::vector<std::array<Point2d, 4>> quads(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double c = std::cos(rotations[static_cast<size_t>(i)]);
        const double s = std::sin(rotations[static_cast<size_t>(i)]);
        const Point2d& center = centers[static_cast<size_t>(i)];
        for (int k = 0; k < 4; ++k) {
            const Point2d& h = half_offsets[static_cast<size_t>(k)];
            quads[static_cast<size_t>(i)][static_cast<size_t>(k)] = {center.x + (c * h.x) - (s * h.y),
                                                                     center.y + (s * h.x) + (c * h.y)};
        }
    }

    const double clamped_inset = std::min(std::max(inset, 0.0), 0.49);
    const double inner = 1.0 - (2.0 * clamped_inset);
    double x_min = quads[0][0].x;
    double x_max = quads[0][0].x;
    double y_min = quads[0][0].y;
    double y_max = quads[0][0].y;
    for (const auto& quad : quads) {
        for (const Point2d& p : quad) {
            x_min = std::min(x_min, p.x);
            x_max = std::max(x_max, p.x);
            y_min = std::min(y_min, p.y);
            y_max = std::max(y_max, p.y);
        }
    }
    double range_x = x_max - x_min;
    double range_y = y_max - y_min;
    if (range_x < 1e-9 || range_y < 1e-9) {
        range_x = std::max(range_x, 1.0);
        range_y = std::max(range_y, 1.0);
    }

    const double scale = inner * std::min(1.0 / range_x, 1.0 / range_y);
    const double offset_x = ((inner - (range_x * scale)) / 2.0) - (x_min * scale);
    const double offset_y = ((inner - (range_y * scale)) / 2.0) - (y_min * scale);

    std::vector<RotatedRect> card_poses(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Point2d& center = centers[static_cast<size_t>(i)];
        card_poses[static_cast<size_t>(i)] = {(center.x * scale) + offset_x, (center.y * scale) + offset_y,
                                              rotations[static_cast<size_t>(i)]};
    }

    std::vector<int> z_orders(static_cast<size_t>(n), 0);
    const std::vector<int> order = occlusion_order(detected_cards);
    for (size_t rank = 0; rank < order.size(); ++rank) {
        z_orders[static_cast<size_t>(order[rank])] = n - 1 - static_cast<int>(rank);
    }

    return CardsArrangement{card_poses, z_orders, common_w * scale, common_h * scale};
}

auto create_session(Ort::Env& env, const std::string& model_path, int device_id) -> Ort::Session {
    Ort::SessionOptions opts;
    if (device_id >= 0) {
        try {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = device_id;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
            // NOLINTNEXTLINE(bugprone-empty-catch) deliberate: fall back to CPU on any CUDA setup failure
        } catch (...) {
        }
    }
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return {env, model_path.c_str(), opts};
}

auto detect_card_corners(const cv::Mat& img_rgb, Ort::Session& session, double conf, double iou, int imgsz)
    -> std::vector<CornerDetection> {
    const LetterboxResult lb = letterbox(img_rgb, imgsz, imgsz);

    const auto plane = static_cast<size_t>(imgsz) * static_cast<size_t>(imgsz);
    std::vector<float> tensor(plane * 3);
    for (int y = 0; y < imgsz; ++y) {
        const uint8_t* row = lb.image.ptr(y);
        for (int x = 0; x < imgsz; ++x) {
            for (int ch = 0; ch < 3; ++ch) {
                tensor[(static_cast<size_t>(ch) * plane) + (static_cast<size_t>(y) * static_cast<size_t>(imgsz)) +
                       static_cast<size_t>(x)] =
                    static_cast<float>(row[(static_cast<size_t>(x) * 3) + static_cast<size_t>(ch)]) / 255.0F;
            }
        }
    }

    const std::array<int64_t, 4> shape = {1, 3, imgsz, imgsz};
    const Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(mem_info, tensor.data(), tensor.size(), shape.data(), shape.size());

    const Ort::AllocatorWithDefaultOptions allocator;
    const Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(0, allocator);
    const char* out_name_c = out_name.get();
    constexpr const char* kInputName = "images";
    auto outputs = session.Run(Ort::RunOptions{}, &kInputName, &input_tensor, 1, &out_name_c, 1);

    const Ort::Value& out = outputs[0];
    const std::vector<int64_t> out_shape = out.GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 3 || out_shape[1] != 17) {
        throw std::runtime_error("unexpected corner model output shape");
    }
    const auto n_anchors = static_cast<size_t>(out_shape[2]);
    const auto* raw = out.GetTensorData<float>();

    std::vector<float> transposed(n_anchors * 17);
    for (size_t c = 0; c < 17; ++c) {
        for (size_t a = 0; a < n_anchors; ++a) {
            transposed[(a * 17) + c] = raw[(c * n_anchors) + a];
        }
    }

    return parse_corner_output(transposed.data(), static_cast<int>(n_anchors), 17, conf, iou, img_rgb.rows,
                               img_rgb.cols, lb.ratio, lb.pad_left, lb.pad_top);
}

auto classify_cards(const cv::Mat& img_rgb, const std::vector<std::array<Point2d, 4>>& corners_list,
                    Ort::Session& session) -> std::vector<ClassProbs> {
    if (corners_list.empty()) {
        return {};
    }

    std::vector<float> batch;
    batch.reserve(corners_list.size() * static_cast<size_t>(kClassCardWidth) * static_cast<size_t>(kClassCardHeight) *
                  3);
    for (const std::array<Point2d, 4>& quad : corners_list) {
        std::optional<cv::Mat> card = warp_card(img_rgb, quad, kClassCardWidth, kClassCardHeight);
        const cv::Mat card_img =
            card.has_value() ? std::move(*card) : cv::Mat::zeros(kClassCardHeight, kClassCardWidth, CV_8UC3);
        append_normalized_card(card_img, batch);
    }

    const auto n = static_cast<int64_t>(corners_list.size());
    const std::array<int64_t, 4> shape = {n, 3, kClassCardHeight, kClassCardWidth};
    const Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(mem_info, batch.data(), batch.size(), shape.data(), shape.size());

    std::vector<Ort::AllocatedStringPtr> holders;
    const std::vector<const char*> output_names = all_output_names(session, holders);
    constexpr const char* kInputName = "input";
    auto outputs =
        session.Run(Ort::RunOptions{}, &kInputName, &input_tensor, 1, output_names.data(), output_names.size());

    // Bind each head's output by its declared ONNX name rather than assuming a fixed
    // position: the graph's internal tensor order isn't guaranteed to match output_names,
    // so binding by name is the only way to stay correct regardless of export ordering.
    const size_t count_i = find_output_index(output_names, "count");
    const size_t color_i = find_output_index(output_names, "color");
    const size_t fill_i = find_output_index(output_names, "fill");
    const size_t shape_i = find_output_index(output_names, "shape");
    const auto* count_data = outputs[count_i].GetTensorData<float>();
    const auto* color_data = outputs[color_i].GetTensorData<float>();
    const auto* fill_data = outputs[fill_i].GetTensorData<float>();
    const auto* shape_data = outputs[shape_i].GetTensorData<float>();

    std::vector<ClassProbs> results(corners_list.size());
    for (size_t i = 0; i < corners_list.size(); ++i) {
        std::array<double, 4> count_logits{};
        std::array<double, 4> color_logits{};
        std::array<double, 4> fill_logits{};
        std::array<double, 4> shape_logits{};
        for (size_t k = 0; k < 4; ++k) {
            count_logits[k] = count_data[(i * 4) + k];
            color_logits[k] = color_data[(i * 4) + k];
            fill_logits[k] = fill_data[(i * 4) + k];
            shape_logits[k] = shape_data[(i * 4) + k];
        }
        results[i].count_probs = softmax4(count_logits);
        results[i].color_probs = softmax4(color_logits);
        results[i].fill_probs = softmax4(fill_logits);
        results[i].shape_probs = softmax4(shape_logits);
    }
    return results;
}

auto detect_cards(const cv::Mat& img_rgb, Ort::Session& corner_session, Ort::Session& class_session, double conf,
                  double iou, int imgsz) -> Detection {
    const std::vector<CornerDetection> detections = detect_card_corners(img_rgb, corner_session, conf, iou, imgsz);

    std::vector<std::array<Point2d, 4>> corners_list;
    corners_list.reserve(detections.size());
    for (const CornerDetection& d : detections) {
        corners_list.push_back(d.corners);
    }
    const std::vector<ClassProbs> classifications = classify_cards(img_rgb, corners_list, class_session);

    std::vector<DetectedCard> detected_cards;
    detected_cards.reserve(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        DetectedCard dc;
        dc.corners = detections[i].corners;
        dc.corner_visibility = detections[i].visibility;
        dc.count_probs = classifications[i].count_probs;
        dc.color_probs = classifications[i].color_probs;
        dc.shape_probs = classifications[i].shape_probs;
        dc.fill_probs = classifications[i].fill_probs;
        detected_cards.push_back(dc);
    }

    std::vector<std::pair<DetectedCard, set_game::Card>> matches;
    std::vector<DetectedCard> matched_cards;
    for (const DetectedCard& dc : detected_cards) {
        if (const std::optional<set_game::Card> card = match_card(dc)) {
            matches.emplace_back(dc, *card);
            matched_cards.push_back(dc);
        }
    }

    CardsArrangement matches_arrangement;
    try {
        matches_arrangement = cards_arrangement(matched_cards);
    } catch (const std::runtime_error&) {
        matches_arrangement = CardsArrangement{};
    }

    return {std::move(detected_cards), std::move(matches), std::move(matches_arrangement)};
}

struct CardDetector::Impl {
    // Declared before the sessions: pimpl members destruct in reverse declaration
    // order, and a Session outliving its Env is a crash risk.
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "setdetect"};
    std::unique_ptr<Ort::Session> corner_session;
    std::unique_ptr<Ort::Session> class_session;
    double conf = 0.25;
    double iou = 0.45;
    int imgsz = 640;
};

CardDetector::CardDetector(const std::string& corner_weights, const std::string& class_weights, int device_id,
                           double conf, double iou, int imgsz, bool warm_up)
    : impl_(std::make_unique<Impl>()) {
    impl_->corner_session = std::make_unique<Ort::Session>(create_session(impl_->env, corner_weights, device_id));
    impl_->class_session = std::make_unique<Ort::Session>(create_session(impl_->env, class_weights, device_id));
    impl_->conf = conf;
    impl_->iou = iou;
    impl_->imgsz = imgsz;
    if (warm_up) {
        warm_up_corner_session(*impl_->corner_session, imgsz);
        warm_up_class_session(*impl_->class_session);
    }
}

CardDetector::~CardDetector() = default;
CardDetector::CardDetector(CardDetector&&) noexcept = default;
auto CardDetector::operator=(CardDetector&&) noexcept -> CardDetector& = default;

auto CardDetector::detect(const cv::Mat& img_rgb) const -> Detection {
    return detect_cards(img_rgb, *impl_->corner_session, *impl_->class_session, impl_->conf, impl_->iou, impl_->imgsz);
}

} // namespace setdetect::card_detection
