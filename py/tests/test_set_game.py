import pytest

from setdetect.set_game.game import Card, Color, Count, Fill, GameState, Shape


@pytest.mark.parametrize(
    "count,color,shape,fill",
    [
        (Count.ONE, Color.RED, Shape.DIAMOND, Fill.OPEN),
        (Count.TWO, Color.GREEN, Shape.OVAL, Fill.STRIPED),
        (Count.THREE, Color.PURPLE, Shape.SQUIGGLE, Fill.SOLID),
    ],
)
def test_card_attributes_and_label(count, color, shape, fill):
    card = Card(count=count, color=color, shape=shape, fill=fill)
    assert card.attributes == (count, color, shape, fill)
    assert card.label == (count.value, color.value, shape.value, fill.value)


def test_card_is_frozen():
    card = Card(Count.ONE, Color.RED, Shape.DIAMOND, Fill.OPEN)
    with pytest.raises(AttributeError):
        card.count = Count.TWO  # type: ignore[misc]


def test_card_equality_by_value():
    a = Card(Count.ONE, Color.RED, Shape.DIAMOND, Fill.OPEN)
    b = Card(Count.ONE, Color.RED, Shape.DIAMOND, Fill.OPEN)
    c = Card(Count.TWO, Color.RED, Shape.DIAMOND, Fill.OPEN)
    assert a == b
    assert a != c


def test_game_state_holds_cards():
    cards = [Card(Count.ONE, Color.RED, Shape.DIAMOND, Fill.OPEN)]
    state = GameState(playing_area=cards)
    assert state.playing_area == cards
