#include <setdetect/set_game/game.hpp>
#include <string>

namespace setdetect::set_game {

auto to_string(Count c) -> std::string {
    switch (c) {
    case Count::ONE:
        return "one";
    case Count::TWO:
        return "two";
    case Count::THREE:
        return "three";
    case Count::OTHER:
        return "other";
    }
    return "other";
}

auto to_string(Color c) -> std::string {
    switch (c) {
    case Color::RED:
        return "red";
    case Color::GREEN:
        return "green";
    case Color::PURPLE:
        return "purple";
    case Color::OTHER:
        return "other";
    }
    return "other";
}

auto to_string(Shape s) -> std::string {
    switch (s) {
    case Shape::DIAMOND:
        return "diamond";
    case Shape::OVAL:
        return "oval";
    case Shape::SQUIGGLE:
        return "squiggle";
    case Shape::OTHER:
        return "other";
    }
    return "other";
}

auto to_string(Fill f) -> std::string {
    switch (f) {
    case Fill::OPEN:
        return "open";
    case Fill::SOLID:
        return "solid";
    case Fill::STRIPED:
        return "striped";
    case Fill::OTHER:
        return "other";
    }
    return "other";
}

auto count_letter(Count c) -> char {
    switch (c) {
    case Count::ONE:
        return '1';
    case Count::TWO:
        return '2';
    case Count::THREE:
        return '3';
    case Count::OTHER:
        return '?';
    }
    return '?';
}

auto color_letter(Color c) -> char {
    switch (c) {
    case Color::RED:
        return 'r';
    case Color::GREEN:
        return 'g';
    case Color::PURPLE:
        return 'p';
    case Color::OTHER:
        return '?';
    }
    return '?';
}

auto shape_letter(Shape s) -> char {
    switch (s) {
    case Shape::DIAMOND:
        return 'd';
    case Shape::OVAL:
        return 'e';
    case Shape::SQUIGGLE:
        return 'q';
    case Shape::OTHER:
        return '?';
    }
    return '?';
}

auto fill_letter(Fill f) -> char {
    switch (f) {
    case Fill::OPEN:
        return 'o';
    case Fill::SOLID:
        return 's';
    case Fill::STRIPED:
        return 't';
    case Fill::OTHER:
        return '?';
    }
    return '?';
}

} // namespace setdetect::set_game
