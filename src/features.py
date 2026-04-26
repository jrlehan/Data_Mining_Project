"""Smaller feature helpers that don't deserve their own module.

This file holds:

* `simplify_action` / `simplify_available_moves` -- collapse the raw
  free-text decision labels into one of {Bet/Raise, Check, Call, Fold,
  Other}.
* `parse_holding` -- pull rank ints out of a 4-character holding like
  'AhKd'.
* `POSITION_VALUES`, `STREET_VALUES` -- lookup tables for encoding
  IP/OOP and Flop/Turn/River as small integers.
* `build_card_onehots` -- one-hot encode every card slot in
  hole+board into 7 * 52 columns.
* `board_aggregate_features` -- suit / rank / connectedness summaries
  of the visible cards.
"""

from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cards import RANK_VALUES, cards_to_strings


# ---------------------------------------------------------------------------
# Action / move simplification
# ---------------------------------------------------------------------------

def simplify_action(action):
    """Collapse a free-text action label into one of 5 canonical buckets."""
    action = action.lower()
    if "bet" in action or "raise" in action:
        return "Bet/Raise"
    if "check" in action:
        return "Check"
    if "call" in action:
        return "Call"
    if "fold" in action:
        return "Fold"
    return "Other"


def simplify_available_moves(moves):
    """Apply `simplify_action` to a comma-joined list of available moves."""
    simplified = set()
    for move in moves.split(","):
        move = move.lower()
        if "bet" in move or "raise" in move:
            simplified.add("Bet/Raise")
        elif "check" in move:
            simplified.add("Check")
        elif "call" in move:
            simplified.add("Call")
        elif "fold" in move:
            simplified.add("Fold")
        else:
            simplified.add("Other")
    return sorted(simplified)


# ---------------------------------------------------------------------------
# Holding / position / street encodings
# ---------------------------------------------------------------------------

def parse_holding(holding):
    """Extract the two rank ints from a 4-char holding like 'AhKd'."""
    return RANK_VALUES[holding[0]], RANK_VALUES[holding[2]]


POSITION_VALUES = {'IP': 1, 'OOP': 0}
STREET_VALUES = {'Flop': 1, 'Turn': 2, 'River': 3}


# ---------------------------------------------------------------------------
# Card one-hots and board aggregates (for the "natural features" model)
# ---------------------------------------------------------------------------

def card_to_idx(card):
    """'As' -> 0..51. Returns -1 if `card` is invalid or missing."""
    if not isinstance(card, str) or len(card) != 2:
        return -1
    try:
        return (RANK_VALUES[card[0]] - 2) * 4 + 'hdcs'.index(card[1])
    except (KeyError, ValueError):
        return -1


def build_card_onehots(df_src):
    """One-hot encode every card slot. Missing cards -> all zeros.

    Produces 7 slots × 52 cards = 364 binary columns: hole0, hole1,
    flop0, flop1, flop2, turn, river. Slots without a card (e.g. turn
    on a Flop row) are encoded as all zeros.
    """
    feats = {}
    slot_lookup = {
        'holding': df_src['holding'].apply(cards_to_strings).values,
        'board_flop': df_src['board_flop'].apply(cards_to_strings).values,
        'board_turn': df_src['board_turn'].apply(cards_to_strings).values,
        'board_river': df_src['board_river'].apply(cards_to_strings).values,
    }

    slots = [
        ('holding', 0, 'hole0'),
        ('holding', 1, 'hole1'),
        ('board_flop', 0, 'flop0'),
        ('board_flop', 1, 'flop1'),
        ('board_flop', 2, 'flop2'),
        ('board_turn', 0, 'turn'),
        ('board_river', 0, 'river'),
    ]

    for col, idx, prefix in slots:
        cards = slot_lookup[col]
        indices = np.array([
            card_to_idx(cards_for_row[idx]) if len(cards_for_row) > idx else -1
            for cards_for_row in cards
        ])
        onehot = np.zeros((len(df_src), 52), dtype=np.int8)
        mask = indices >= 0
        onehot[np.where(mask)[0], indices[mask]] = 1
        for c in range(52):
            feats[f'{prefix}_c{c}'] = onehot[:, c]
    return pd.DataFrame(feats, index=df_src.index)


def board_aggregate_features(df_src):
    """Summary stats over the visible cards: suits, rank dups, connectedness."""
    recs = []
    for _, row in tqdm(df_src.iterrows(), total=len(df_src),
                       desc="Board aggregates"):
        all_cards = cards_to_strings(row['holding'])
        all_cards += cards_to_strings(row['board_flop'])
        if row['evaluation_at'] in ('Turn', 'River'):
            all_cards += cards_to_strings(row['board_turn'])
        if row['evaluation_at'] == 'River':
            all_cards += cards_to_strings(row['board_river'])

        ranks = [RANK_VALUES[c[0]] for c in all_cards]
        suits = [c[1] for c in all_cards]
        suit_counts = Counter(suits)
        rank_counts = Counter(ranks)

        max_suit = max(suit_counts.values()) if suit_counts else 0
        count_dist = sorted(rank_counts.values(), reverse=True)
        n_pairs = sum(1 for c in count_dist if c == 2)
        n_trips = sum(1 for c in count_dist if c == 3)
        n_quads = sum(1 for c in count_dist if c == 4)
        rank_range = max(ranks) - min(ranks) if ranks else 0
        unique_ranks = len(set(ranks))

        rank_set = set(ranks)
        max_straight_run = 0
        for lo in range(2, 11):
            window = set(range(lo, lo + 5))
            max_straight_run = max(max_straight_run, len(window & rank_set))
        wheel = {14, 2, 3, 4, 5}
        max_straight_run = max(max_straight_run, len(wheel & rank_set))

        recs.append({
            'max_suit_count': max_suit,
            'n_pairs': n_pairs,
            'n_trips': n_trips,
            'n_quads': n_quads,
            'rank_range': rank_range,
            'n_unique_ranks': unique_ranks,
            'max_straight_run': max_straight_run,
        })
    return pd.DataFrame.from_records(recs, index=df_src.index)
