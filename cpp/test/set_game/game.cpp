#include <gtest/gtest.h>
#include <setdetect/set_game/game.hpp>

namespace sg = setdetect::set_game;

TEST(SetGameTest, CountToStringAndLetter) {
    EXPECT_EQ(sg::to_string(sg::Count::ONE), "one");
    EXPECT_EQ(sg::to_string(sg::Count::TWO), "two");
    EXPECT_EQ(sg::to_string(sg::Count::THREE), "three");
    EXPECT_EQ(sg::to_string(sg::Count::OTHER), "other");

    EXPECT_EQ(sg::count_letter(sg::Count::ONE), '1');
    EXPECT_EQ(sg::count_letter(sg::Count::TWO), '2');
    EXPECT_EQ(sg::count_letter(sg::Count::THREE), '3');
    EXPECT_EQ(sg::count_letter(sg::Count::OTHER), '?');
}

TEST(SetGameTest, ColorToStringAndLetter) {
    EXPECT_EQ(sg::to_string(sg::Color::RED), "red");
    EXPECT_EQ(sg::to_string(sg::Color::GREEN), "green");
    EXPECT_EQ(sg::to_string(sg::Color::PURPLE), "purple");
    EXPECT_EQ(sg::to_string(sg::Color::OTHER), "other");

    EXPECT_EQ(sg::color_letter(sg::Color::RED), 'r');
    EXPECT_EQ(sg::color_letter(sg::Color::GREEN), 'g');
    EXPECT_EQ(sg::color_letter(sg::Color::PURPLE), 'p');
    EXPECT_EQ(sg::color_letter(sg::Color::OTHER), '?');
}

TEST(SetGameTest, ShapeToStringAndLetter) {
    EXPECT_EQ(sg::to_string(sg::Shape::DIAMOND), "diamond");
    EXPECT_EQ(sg::to_string(sg::Shape::OVAL), "oval");
    EXPECT_EQ(sg::to_string(sg::Shape::SQUIGGLE), "squiggle");
    EXPECT_EQ(sg::to_string(sg::Shape::OTHER), "other");

    EXPECT_EQ(sg::shape_letter(sg::Shape::DIAMOND), 'd');
    EXPECT_EQ(sg::shape_letter(sg::Shape::OVAL), 'e');
    EXPECT_EQ(sg::shape_letter(sg::Shape::SQUIGGLE), 'q');
    EXPECT_EQ(sg::shape_letter(sg::Shape::OTHER), '?');
}

TEST(SetGameTest, FillToStringAndLetter) {
    EXPECT_EQ(sg::to_string(sg::Fill::OPEN), "open");
    EXPECT_EQ(sg::to_string(sg::Fill::SOLID), "solid");
    EXPECT_EQ(sg::to_string(sg::Fill::STRIPED), "striped");
    EXPECT_EQ(sg::to_string(sg::Fill::OTHER), "other");

    EXPECT_EQ(sg::fill_letter(sg::Fill::OPEN), 'o');
    EXPECT_EQ(sg::fill_letter(sg::Fill::SOLID), 's');
    EXPECT_EQ(sg::fill_letter(sg::Fill::STRIPED), 't');
    EXPECT_EQ(sg::fill_letter(sg::Fill::OTHER), '?');
}
