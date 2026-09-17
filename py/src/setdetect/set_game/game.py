from dataclasses import dataclass
from enum import Enum


class Count(Enum):
    ONE = "one"
    TWO = "two"
    THREE = "three"


class Color(Enum):
    RED = "red"
    GREEN = "green"
    PURPLE = "purple"


class Shape(Enum):
    DIAMOND = "diamond"
    OVAL = "oval"
    SQUIGGLE = "squiggle"


class Fill(Enum):
    OPEN = "open"
    STRIPED = "striped"
    SOLID = "solid"


@dataclass(frozen=True)
class Card:
    count: Count
    color: Color
    shape: Shape
    fill: Fill

    @property
    def attributes(self) -> tuple[Count, Color, Shape, Fill]:
        return (self.count, self.color, self.shape, self.fill)

    @property
    def label(self) -> tuple[str, str, str, str]:
        return (
            self.count.value,
            self.color.value,
            self.shape.value,
            self.fill.value,
        )


@dataclass
class GameState:
    playing_area: list[Card]
