#include <apps/path_util.hpp>
#include <apps/stats.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace sa = setdetect::apps;

TEST(ExpandTildeTest, ExpandsLeadingTildeWhenHomeIsSet) {
    ASSERT_NE(std::getenv("HOME"), nullptr);
    const std::string home = std::getenv("HOME");
    EXPECT_EQ(sa::expand_tilde("~/data/set-cards"), home + "/data/set-cards");
}

TEST(ExpandTildeTest, LeavesPathWithoutLeadingTildeUnchanged) {
    EXPECT_EQ(sa::expand_tilde("/abs/path"), "/abs/path");
    EXPECT_EQ(sa::expand_tilde("relative/path"), "relative/path");
}

TEST(ExpandTildeTest, EmptyStringUnchanged) { EXPECT_EQ(sa::expand_tilde(""), ""); }

TEST(SummarizeTest, ComputesMeanAndPopulationStddev) {
    const std::vector<double> samples = {1.0, 2.0, 3.0, 4.0, 5.0};
    const sa::Stats stats = sa::summarize(samples);
    EXPECT_DOUBLE_EQ(stats.mean, 3.0);
    EXPECT_NEAR(stats.stddev, std::sqrt(2.0), 1e-12);
}

TEST(SummarizeTest, ConstantSamplesHaveZeroStddev) {
    const std::vector<double> samples = {7.0, 7.0, 7.0};
    const sa::Stats stats = sa::summarize(samples);
    EXPECT_DOUBLE_EQ(stats.mean, 7.0);
    EXPECT_DOUBLE_EQ(stats.stddev, 0.0);
}
