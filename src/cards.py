"""Shared card-parsing primitives used across feature modules.

A "card string" in this dataset is two characters: a rank character
('2'..'9', 'T', 'J', 'Q', 'K', 'A') followed by a suit character
('h', 'd', 'c', 's'). Multiple cards are concatenated with no separator,
e.g. 'AhKd' or 'JcJh4s'.
"""

RANK_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
}

ALL_SUITS = ['h', 'd', 'c', 's']
ALL_RANKS = list(RANK_VALUES.keys())
FULL_DECK = [r + s for r in ALL_RANKS for s in ALL_SUITS]

HAND_RANKS = {
    'high_card': 1,
    'pair': 2,
    'two_pair': 3,
    'three_of_a_kind': 4,
    'straight': 5,
    'flush': 6,
    'full_house': 7,
    'four_of_a_kind': 8,
    'straight_flush': 9,
    'royal_flush': 10,
}

CATEGORY_NAMES = {v: k for k, v in HAND_RANKS.items()}


def parse_cards(card_string):
    """Parse a concatenated card string into [(rank_int, suit_char), ...].

    Empty / non-string input returns an empty list.
    """
    if not isinstance(card_string, str) or card_string == '':
        return []
    return [(RANK_VALUES[card_string[i]], card_string[i + 1])
            for i in range(0, len(card_string), 2)]


def cards_to_strings(card_string):
    """Split a concatenated card string into ['Ah', 'Kd', ...].

    Empty / non-string input returns an empty list.
    """
    if not isinstance(card_string, str) or card_string == '':
        return []
    return [card_string[i:i + 2] for i in range(0, len(card_string), 2)]
