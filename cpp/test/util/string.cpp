#include <gtest/gtest.h>
#include <setdetect/util/string.hpp>
#include <string>
#include <vector>

namespace su = setdetect::util::string;

TEST(StringJoinTest, EmptyVectorReturnsEmptyString) { EXPECT_EQ(su::join({}, ", "), ""); }

TEST(StringJoinTest, SingleElementReturnsItselfUnjoined) { EXPECT_EQ(su::join({"solo"}, ", "), "solo"); }

TEST(StringJoinTest, MultipleElementsAreJoinedWithDelimiter) {
    const std::vector<std::string> parts = {"one", "two", "three"};
    EXPECT_EQ(su::join(parts, ", "), "one, two, three");
}

TEST(StringJoinTest, DelimiterCanShareCharactersWithElements) {
    const std::vector<std::string> parts = {"a", "b", "c"};
    EXPECT_EQ(su::join(parts, "a"), "aabac");
}

TEST(StringJoinTest, EmptyDelimiterConcatenatesDirectly) {
    const std::vector<std::string> parts = {"foo", "bar"};
    EXPECT_EQ(su::join(parts, ""), "foobar");
}
