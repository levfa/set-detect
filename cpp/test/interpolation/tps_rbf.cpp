#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <setdetect/interpolation/tps_rbf.hpp>
#include <vector>

namespace si = setdetect::interpolation;

namespace {

auto v2(double x, double y) -> Eigen::VectorXd {
    Eigen::VectorXd v(2);
    v << x, y;
    return v;
}

auto v1(double x) -> Eigen::VectorXd {
    Eigen::VectorXd v(1);
    v << x;
    return v;
}

} // namespace

TEST(TpsRbfTest, NodesInterpolateExactly2DTo1D) {
    std::vector<Eigen::VectorXd> inputs = {v2(0.0, 0.0), v2(1.0, 0.0), v2(0.0, 1.0), v2(1.0, 1.0)};
    std::vector<Eigen::VectorXd> outputs = {v1(1.0), v1(2.0), v1(3.0), v1(4.0)};

    si::ThinPlateSplineRBF const rbf(inputs, outputs);

    for (size_t i = 0; i < inputs.size(); ++i) {
        Eigen::VectorXd result = rbf.interpolate(inputs[i]);
        EXPECT_NEAR(result(0), outputs[i](0), 1e-10) << "node " << i << " not interpolated exactly";
    }
}

TEST(TpsRbfTest, KnownValues2DTo1D) {
    std::vector<Eigen::VectorXd> const inputs = {v2(0.0, 0.0), v2(1.0, 0.0), v2(0.0, 1.0), v2(1.0, 1.0)};
    std::vector<Eigen::VectorXd> const outputs = {v1(1.0), v1(2.0), v1(3.0), v1(4.0)};

    si::ThinPlateSplineRBF const rbf(inputs, outputs);

    // Reference values generated with scipy.interpolate.RBFInterpolator(kernel=
    // "thin_plate_spline", smoothing=0.0) -- these 4 corner outputs are exactly the
    // affine function 1 + x + 2y, and the degree-1-polynomial-aware interpolant
    // reproduces that affine function exactly.
    struct Query {
        double x, y, expected;
    };
    std::array<Query, 4> const queries = {
        Query{0.5, 0.0, 1.5},
        Query{0.0, 0.5, 2.0},
        Query{0.5, 0.5, 2.5},
        Query{0.25, 0.75, 2.75},
    };

    for (const auto& q : queries) {
        Eigen::VectorXd result = rbf.interpolate(v2(q.x, q.y));
        EXPECT_NEAR(result(0), q.expected, 1e-10) << "query (" << q.x << ", " << q.y << ")";
    }
}

TEST(TpsRbfTest, KnownValues2DTo2D) {
    std::vector<Eigen::VectorXd> const inputs = {v2(0.0, 0.0), v2(1.0, 0.0), v2(0.0, 1.0), v2(1.0, 1.0)};

    auto make_out = [](double a, double b) {
        Eigen::VectorXd v(2);
        v << a, b;
        return v;
    };
    std::vector<Eigen::VectorXd> const outputs = {make_out(1.0, 0.5), make_out(2.0, 1.5), make_out(3.0, 2.5),
                                                  make_out(4.0, 3.5)};

    si::ThinPlateSplineRBF const rbf(inputs, outputs);

    // Reference values generated with scipy.interpolate.RBFInterpolator(kernel=
    // "thin_plate_spline", smoothing=0.0).
    struct Query {
        double x, y;
        double e0, e1;
    };
    std::array<Query, 4> const queries = {
        Query{0.5, 0.0, 1.5, 1.0},
        Query{0.0, 0.5, 2.0, 1.5},
        Query{0.5, 0.5, 2.5, 2.0},
        Query{0.25, 0.75, 2.75, 2.25},
    };

    for (const auto& q : queries) {
        Eigen::VectorXd result = rbf.interpolate(v2(q.x, q.y));
        ASSERT_EQ(result.size(), 2);
        EXPECT_NEAR(result(0), q.e0, 1e-10) << "query (" << q.x << ", " << q.y << ") dim 0";
        EXPECT_NEAR(result(1), q.e1, 1e-10) << "query (" << q.x << ", " << q.y << ") dim 1";
    }
}

TEST(TpsRbfTest, NodesInterpolateExactly1D) {
    std::vector<Eigen::VectorXd> inputs = {v1(0.0), v1(1.0), v1(3.0), v1(4.0)};
    std::vector<Eigen::VectorXd> outputs = {v1(0.0), v1(2.0), v1(-1.0), v1(3.0)};

    si::ThinPlateSplineRBF const rbf(inputs, outputs);

    for (size_t i = 0; i < inputs.size(); ++i) {
        Eigen::VectorXd result = rbf.interpolate(inputs[i]);
        EXPECT_NEAR(result(0), outputs[i](0), 1e-10) << "node " << i << " not interpolated exactly";
    }
}

TEST(TpsRbfTest, KnownValues1D) {
    std::vector<Eigen::VectorXd> const inputs = {v1(0.0), v1(1.0), v1(3.0), v1(4.0)};
    std::vector<Eigen::VectorXd> const outputs = {v1(0.0), v1(2.0), v1(-1.0), v1(3.0)};

    si::ThinPlateSplineRBF const rbf(inputs, outputs);

    // Reference values generated with scipy.interpolate.RBFInterpolator(kernel=
    // "thin_plate_spline", smoothing=0.0).
    struct Query {
        double x;
        double expected;
    };
    std::array<Query, 4> const queries = {
        Query{0.5, 1.2312265242096392},
        Query{1.5, 1.3538070797660193},
        Query{2.0, 0.1968597201155956},
        Query{3.5, 0.7361250743936261},
    };

    for (const auto& q : queries) {
        Eigen::VectorXd result = rbf.interpolate(v1(q.x));
        EXPECT_NEAR(result(0), q.expected, 1e-10) << "query " << q.x;
    }
}

// Regression test for a real bug: without the polynomial term, a pure-translation
// field (the single most common case in real camera tracking) is not represented
// correctly once queried outside the training nodes' convex hull -- confirmed to be
// off by hundreds of units before this was fixed. scipy.interpolate.RBFInterpolator
// recovers the exact translation even far outside the training data; this must too.
TEST(TpsRbfTest, TranslationFieldExtrapolatesExactly) {
    std::vector<Eigen::VectorXd> const inputs = {v2(0.0, 0.0),   v2(10.0, 0.0), v2(0.0, 10.0),
                                                 v2(10.0, 10.0), v2(5.0, 5.0),  v2(3.0, 8.0)};
    std::vector<Eigen::VectorXd> const outputs(inputs.size(), v2(2.0, -1.0));

    si::ThinPlateSplineRBF const rbf(inputs, outputs, /*smoothing=*/1.0);

    for (const auto& q : {v2(100.0, 100.0), v2(-50.0, -50.0)}) {
        Eigen::VectorXd result = rbf.interpolate(q);
        EXPECT_NEAR(result(0), 2.0, 1e-9) << "query (" << q(0) << ", " << q(1) << ") dim 0";
        EXPECT_NEAR(result(1), -1.0, 1e-9) << "query (" << q(0) << ", " << q(1) << ") dim 1";
    }
}

TEST(TpsRbfTest, SmoothingRegularizesAwayFromExactFit) {
    std::vector<Eigen::VectorXd> const inputs = {v2(0.0, 0.0), v2(1.0, 0.0), v2(0.0, 1.0), v2(1.0, 1.0)};
    std::vector<Eigen::VectorXd> const outputs = {v1(1.0), v1(2.0), v1(3.0), v1(4.1)};

    si::ThinPlateSplineRBF const rbf_exact(inputs, outputs, /*smoothing=*/0.0);
    si::ThinPlateSplineRBF const rbf_smoothed(inputs, outputs, /*smoothing=*/1.0);

    // Reference values generated with scipy.interpolate.RBFInterpolator(kernel=
    // "thin_plate_spline", smoothing=0.0 and 1.0 respectively).
    EXPECT_NEAR(rbf_exact.interpolate(v2(1.0, 1.0))(0), 4.1, 1e-9);
    EXPECT_NEAR(rbf_smoothed.interpolate(v2(1.0, 1.0))(0), 4.0852345973, 1e-8);
}
