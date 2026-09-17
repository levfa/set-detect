#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace setdetect::set_game {

enum class Count : std::uint8_t { ONE, TWO, THREE, OTHER };
enum class Color : std::uint8_t { RED, GREEN, PURPLE, OTHER };
enum class Shape : std::uint8_t { DIAMOND, OVAL, SQUIGGLE, OTHER };
enum class Fill : std::uint8_t { OPEN, SOLID, STRIPED, OTHER };

auto to_string(Count c) -> std::string;
auto to_string(Color c) -> std::string;
auto to_string(Shape s) -> std::string;
auto to_string(Fill f) -> std::string;

auto count_letter(Count c) -> char;
auto color_letter(Color c) -> char;
auto shape_letter(Shape s) -> char;
auto fill_letter(Fill f) -> char;

struct Card {
    Count count;
    Color color;
    Shape shape;
    Fill fill;
};

struct PredCard {
    std::array<std::array<float, 2>, 4> corners;
    std::array<float, 4> corner_confidence;
    std::array<float, 4> count_probs;
    std::array<float, 4> color_probs;
    std::array<float, 4> shape_probs;
    std::array<float, 4> fill_probs;
    std::optional<Card> card;
};

struct PredGameState {
    std::vector<PredCard> playing_area;
};

} // namespace setdetect::set_game
