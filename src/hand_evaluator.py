"""5-card poker hand evaluator.

Evaluates a 5-card hand into a tuple (category, tiebreakers...) that
sorts correctly against any other 5-card tuple, so two hands can be
ranked with a plain Python `>` comparison.

Also exposes `best_hand_strength`, which finds the best 5-card hand
inside a 5-7 card pool (hole cards + visible board) using
`itertools.combinations`.

Run this module directly (`python -m src.hand_evaluator`) to execute
the self-tests, including a check that the evaluator produces exactly
7,462 distinct equivalence classes -- the theoretical count.
"""

from collections import Counter
from itertools import combinations

from .cards import HAND_RANKS, parse_cards


def evaluate_5_cards(cards):
    """Return (category, tiebreakers...) for a list of 5 (rank, suit) tuples."""
    ranks = sorted([c[0] for c in cards], reverse=True)
    suits = [c[1] for c in cards]

    is_flush = len(set(suits)) == 1

    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = None
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [14, 5, 4, 3, 2]:  # wheel: Ace plays low
            is_straight = True
            straight_high = 5

    # Sort rank counts by (count desc, rank desc) so the "leading" rank
    # (the one with the most copies, ties broken by rank) comes first.
    # e.g. KKK22 -> [(13, 3), (2, 2)]; AAKK7 -> [(14, 2), (13, 2), (7, 1)]
    count_rank = sorted(Counter(ranks).items(), key=lambda x: (-x[1], -x[0]))
    counts = [c for _, c in count_rank]
    rank_order = [r for r, _ in count_rank]

    if is_straight and is_flush:
        if straight_high == 14:
            return (HAND_RANKS['royal_flush'],)
        return (HAND_RANKS['straight_flush'], straight_high)
    if counts[0] == 4:
        return (HAND_RANKS['four_of_a_kind'], rank_order[0], rank_order[1])
    if counts == [3, 2]:
        return (HAND_RANKS['full_house'], rank_order[0], rank_order[1])
    if is_flush:
        return (HAND_RANKS['flush'], *ranks)
    if is_straight:
        return (HAND_RANKS['straight'], straight_high)
    if counts[0] == 3:
        return (HAND_RANKS['three_of_a_kind'],
                rank_order[0], rank_order[1], rank_order[2])
    if counts == [2, 2, 1]:
        return (HAND_RANKS['two_pair'],
                rank_order[0], rank_order[1], rank_order[2])
    if counts[0] == 2:
        return (HAND_RANKS['pair'],
                rank_order[0], rank_order[1], rank_order[2], rank_order[3])
    return (HAND_RANKS['high_card'], *ranks)


def best_hand_strength(holding, flop, turn, river, street):
    """Find the best 5-card hand for the player at the given street.

    Returns a sortable tuple from `evaluate_5_cards`, or None if there
    aren't enough visible cards to form a 5-card hand (shouldn't happen
    on Flop or later, but we guard for it).
    """
    cards = parse_cards(holding) + parse_cards(flop)
    if street == 'Turn':
        cards += parse_cards(turn)
    elif street == 'River':
        cards += parse_cards(turn) + parse_cards(river)

    if len(cards) < 5:
        return None
    if len(cards) == 5:
        return evaluate_5_cards(cards)
    return max(evaluate_5_cards(list(combo)) for combo in combinations(cards, 5))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_hand_evaluator():
    """Sanity tests for the evaluator. Run via `python -m src.hand_evaluator`."""

    # ---- Test 1: known category assignments ----
    test_cases = [
        ([(14, 'h'), (13, 'h'), (12, 'h'), (11, 'h'), (10, 'h')], 'royal_flush'),
        ([(9, 's'), (8, 's'), (7, 's'), (6, 's'), (5, 's')], 'straight_flush'),
        ([(14, 'c'), (5, 'c'), (4, 'c'), (3, 'c'), (2, 'c')], 'straight_flush'),  # wheel SF
        ([(7, 'h'), (7, 'd'), (7, 'c'), (7, 's'), (2, 'h')], 'four_of_a_kind'),
        ([(10, 'h'), (10, 'd'), (10, 'c'), (4, 's'), (4, 'h')], 'full_house'),
        ([(14, 'h'), (10, 'h'), (7, 'h'), (5, 'h'), (2, 'h')], 'flush'),
        ([(9, 's'), (8, 'd'), (7, 'c'), (6, 'h'), (5, 's')], 'straight'),
        ([(14, 's'), (5, 'd'), (4, 'c'), (3, 'h'), (2, 's')], 'straight'),  # wheel
        ([(14, 's'), (13, 'd'), (12, 'c'), (11, 'h'), (10, 's')], 'straight'),  # broadway
        ([(8, 'h'), (8, 'd'), (8, 'c'), (5, 's'), (2, 'h')], 'three_of_a_kind'),
        ([(13, 'h'), (13, 'd'), (7, 'c'), (7, 's'), (2, 'h')], 'two_pair'),
        ([(9, 'h'), (9, 'd'), (7, 'c'), (5, 's'), (2, 'h')], 'pair'),
        ([(14, 'h'), (11, 'd'), (8, 'c'), (5, 's'), (2, 'h')], 'high_card'),
    ]
    for cards, expected in test_cases:
        result = evaluate_5_cards(cards)
        assert result[0] == HAND_RANKS[expected], \
            f"Expected {expected}, got category {result[0]} for {cards}"
    print(f"✓ All {len(test_cases)} category tests passed")

    # ---- Test 2: tiebreakers ----
    ordering_cases = [
        ([(14, 'h'), (14, 'd'), (7, 'c'), (5, 's'), (2, 'h')],   # higher pair
         [(13, 'h'), (13, 'd'), (7, 'c'), (5, 's'), (2, 'h')]),
        ([(9, 'h'), (9, 'd'), (14, 'c'), (5, 's'), (2, 'h')],    # better kicker
         [(9, 'h'), (9, 'd'), (13, 'c'), (5, 's'), (2, 'h')]),
        ([(14, 'h'), (10, 'h'), (7, 'h'), (5, 'h'), (2, 'h')],   # higher flush
         [(13, 'h'), (10, 'h'), (7, 'h'), (5, 'h'), (2, 'h')]),
        ([(6, 's'), (5, 'd'), (4, 'c'), (3, 'h'), (2, 's')],     # 6-high beats wheel
         [(14, 's'), (5, 'd'), (4, 'c'), (3, 'h'), (2, 's')]),
        ([(14, 's'), (13, 'd'), (12, 'c'), (11, 'h'), (10, 's')],  # broadway beats 9-high
         [(9, 's'), (8, 'd'), (7, 'c'), (6, 'h'), (5, 's')]),
        ([(6, 's'), (5, 's'), (4, 's'), (3, 's'), (2, 's')],     # any SF beats any quads
         [(14, 'h'), (14, 'd'), (14, 'c'), (14, 's'), (13, 'h')]),
    ]
    for stronger, weaker in ordering_cases:
        s = evaluate_5_cards(stronger)
        w = evaluate_5_cards(weaker)
        assert s > w, f"Expected {stronger} > {weaker}, got {s} vs {w}"
    print(f"✓ All {len(ordering_cases)} ordering tests passed")

    # ---- Test 3: distinct equivalence classes ----
    # Theoretically there are exactly 7,462 distinct 5-card poker hands.
    all_tuples = set()

    for rank_combo in combinations(range(2, 15), 5):
        mixed = [(r, s) for r, s in zip(rank_combo, ['h', 'd', 'c', 's', 'h'])]
        all_tuples.add(evaluate_5_cards(mixed))
        suited = [(r, 'h') for r in rank_combo]
        all_tuples.add(evaluate_5_cards(suited))

    for pair_r in range(2, 15):
        for kickers in combinations([r for r in range(2, 15) if r != pair_r], 3):
            hand = [(pair_r, 'h'), (pair_r, 'd')] + [(k, 'c') for k in kickers]
            hand = [(r, s) for (r, _), s in zip(hand, ['h', 'd', 'c', 's', 'h'])]
            all_tuples.add(evaluate_5_cards(hand))

    for pairs in combinations(range(2, 15), 2):
        for kicker in [r for r in range(2, 15) if r not in pairs]:
            hand = [(pairs[0], 'h'), (pairs[0], 'd'),
                    (pairs[1], 'c'), (pairs[1], 's'), (kicker, 'h')]
            all_tuples.add(evaluate_5_cards(hand))

    for trip_r in range(2, 15):
        for kickers in combinations([r for r in range(2, 15) if r != trip_r], 2):
            hand = [(trip_r, 'h'), (trip_r, 'd'), (trip_r, 'c'),
                    (kickers[0], 's'), (kickers[1], 'h')]
            all_tuples.add(evaluate_5_cards(hand))

    for trip_r in range(2, 15):
        for pair_r in range(2, 15):
            if pair_r == trip_r:
                continue
            hand = [(trip_r, 'h'), (trip_r, 'd'), (trip_r, 'c'),
                    (pair_r, 's'), (pair_r, 'h')]
            all_tuples.add(evaluate_5_cards(hand))

    for quad_r in range(2, 15):
        for kicker in [r for r in range(2, 15) if r != quad_r]:
            hand = [(quad_r, 'h'), (quad_r, 'd'), (quad_r, 'c'),
                    (quad_r, 's'), (kicker, 'h')]
            all_tuples.add(evaluate_5_cards(hand))

    print(f"Distinct hand-strength tuples found: {len(all_tuples)}")
    assert len(all_tuples) == 7462, f"Expected 7462, got {len(all_tuples)}"
    print("✓ Distinct-class count matches theoretical 7,462")


if __name__ == "__main__":
    _test_hand_evaluator()
