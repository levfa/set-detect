import typing as tp

import numpy as np

from setdetect.data.card_cutouts import MaskedCard, MaskedCardDataset

CardProvider = tp.Callable[[int, np.random.Generator], list[MaskedCard]]
ScenarioFactory = tp.Callable[[MaskedCardDataset], CardProvider]


def duplicate_card_provider(dataset: MaskedCardDataset) -> CardProvider:
    """Card provider for a board with one card identity placed twice.

    Draws ``num_cards - 1`` distinct cards as usual, then re-inserts a copy of one of them
    at a random position, so callers see the normal ``card_provider`` signature and get back
    exactly ``num_cards`` cards with exactly one repeated identity.
    """

    def _provider(num_cards: int, rng: np.random.Generator) -> list[MaskedCard]:
        if num_cards < 2:
            raise ValueError(f"duplicate-card scenario needs at least 2 cards, got {num_cards}")
        cards = dataset.random_cards(num_cards - 1, rng)
        dup = cards[int(rng.integers(len(cards)))]
        cards.insert(int(rng.integers(len(cards) + 1)), dup)
        return cards

    return _provider


SCENARIOS: dict[str, ScenarioFactory] = {
    "duplicate-card": duplicate_card_provider,
}
