"""Best hand the player can still reach by the river.

`max_future_strength_fast` answers: given the player's hole cards and
the visible board, what is the strongest 5-card hand they could
*possibly* end up with after all remaining board cards are dealt?

The naive approach -- enumerate every possible completion of the board
and run the evaluator on each -- is far too slow for 500k rows. Instead
we walk down the hand-rank ladder (royal flush, straight flush, quads,
full house, ...) and for each category check whether enough cards of
the right ranks/suits could plausibly come on remaining board streets.
The first category we can reach is the answer.
"""

from collections import Counter
from itertools import combinations

from .cards import ALL_SUITS, HAND_RANKS, RANK_VALUES, cards_to_strings
from .hand_evaluator import evaluate_5_cards


def max_future_strength_fast(holding, flop, turn, river, street):
    """Best 5-card hand category the player can still reach.

    Returns a sortable tuple in the same shape as `evaluate_5_cards`.
    """
    known = cards_to_strings(holding) + cards_to_strings(flop)
    if street == 'Turn':
        known += cards_to_strings(turn)
    elif street == 'River':
        known += cards_to_strings(turn) + cards_to_strings(river)

    cards_needed = 7 - len(known)

    parsed = [(RANK_VALUES[c[0]], c[1]) for c in known]
    ranks = [r for r, _ in parsed]
    rank_counts = Counter(ranks)

    if cards_needed == 0:
        return max(evaluate_5_cards(list(combo))
                   for combo in combinations(parsed, 5))

    # ---- Royal flush ----
    for suit in ALL_SUITS:
        have = {r for r, s in parsed if s == suit}
        need = {14, 13, 12, 11, 10} - have
        if len(need) <= cards_needed:
            return (HAND_RANKS['royal_flush'],)

    # ---- Straight flush ----
    best_sf_high = 0
    for suit in ALL_SUITS:
        have = {r for r, s in parsed if s == suit}
        for high in range(14, 5, -1):
            needed_ranks = set(range(high - 4, high + 1))
            missing = needed_ranks - have
            if len(missing) <= cards_needed:
                best_sf_high = max(best_sf_high, high)
                break
        if 14 in have or 14 not in have:
            wheel = {14, 2, 3, 4, 5}
            if len(wheel - have) <= cards_needed and best_sf_high == 0:
                best_sf_high = max(best_sf_high, 5)
    if best_sf_high > 0:
        return (HAND_RANKS['straight_flush'], best_sf_high)

    # ---- Four of a kind ----
    for r in range(14, 1, -1):
        if 4 - rank_counts.get(r, 0) <= cards_needed:
            other_ranks = [rr for rr in ranks if rr != r]
            if cards_needed > (4 - rank_counts.get(r, 0)):
                kicker = 14
            else:
                kicker = max(other_ranks) if other_ranks else 2
            return (HAND_RANKS['four_of_a_kind'], r, kicker)

    # ---- Full house ----
    best_fh = None
    for trip_r in range(14, 1, -1):
        trip_need = max(0, 3 - rank_counts.get(trip_r, 0))
        if trip_need > cards_needed:
            continue
        remaining_draws = cards_needed - trip_need
        for pair_r in range(14, 1, -1):
            if pair_r == trip_r:
                continue
            pair_need = max(0, 2 - rank_counts.get(pair_r, 0))
            if pair_need <= remaining_draws:
                best_fh = (HAND_RANKS['full_house'], trip_r, pair_r)
                break
        if best_fh:
            break

    # ---- Flush ----
    best_flush = None
    for suit in ALL_SUITS:
        have = sorted([r for r, s in parsed if s == suit], reverse=True)
        need = max(0, 5 - len(have))
        if need <= cards_needed:
            drawn_ranks = []
            used = set(have)
            for r in range(14, 1, -1):
                if len(drawn_ranks) >= need:
                    break
                if r not in used:
                    drawn_ranks.append(r)
                    used.add(r)
            flush_ranks = sorted(have + drawn_ranks, reverse=True)[:5]
            candidate = (HAND_RANKS['flush'], *flush_ranks)
            if best_flush is None or candidate > best_flush:
                best_flush = candidate

    if best_fh and best_flush:
        return max(best_fh, best_flush)
    if best_fh:
        return best_fh
    if best_flush:
        return best_flush

    # ---- Straight ----
    rank_set = set(ranks)
    best_straight_high = 0
    for high in range(14, 5, -1):
        needed_ranks = set(range(high - 4, high + 1))
        if len(needed_ranks - rank_set) <= cards_needed:
            best_straight_high = high
            break
    if best_straight_high == 0:
        wheel = {14, 2, 3, 4, 5}
        if len(wheel - rank_set) <= cards_needed:
            best_straight_high = 5
    if best_straight_high > 0:
        return (HAND_RANKS['straight'], best_straight_high)

    # ---- Three of a kind ----
    for r in range(14, 1, -1):
        if 3 - rank_counts.get(r, 0) <= cards_needed:
            needed_for_trip = max(0, 3 - rank_counts.get(r, 0))
            spare = cards_needed - needed_for_trip
            other = sorted([rr for rr in ranks if rr != r], reverse=True)
            if spare >= 2:
                kickers = (other + [14, 13])[:2]
            elif spare >= 1:
                kickers = (other + [14])[:2]
            else:
                kickers = other[:2]
            kickers = (kickers + [2, 2])[:2]
            return (HAND_RANKS['three_of_a_kind'], r, kickers[0], kickers[1])

    # ---- Two pair ----
    best_tp = None
    for high_r in range(14, 1, -1):
        high_need = max(0, 2 - rank_counts.get(high_r, 0))
        if high_need > cards_needed:
            continue
        for low_r in range(high_r - 1, 1, -1):
            low_need = max(0, 2 - rank_counts.get(low_r, 0))
            if high_need + low_need <= cards_needed:
                best_tp = (HAND_RANKS['two_pair'], high_r, low_r, 14)
                break
        if best_tp:
            break
    if best_tp:
        return best_tp

    # ---- Pair ----
    for r in range(14, 1, -1):
        if 2 - rank_counts.get(r, 0) <= cards_needed:
            return (HAND_RANKS['pair'], r, 14, 13, 12)

    return (HAND_RANKS['high_card'], 14, 13, 12, 11, 10)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_max_future_strength():
    """Sanity tests for the max-future routine.

    Run via `python -m src.max_strength`. These are data-independent
    fixtures — the data-dependent invariants (e.g. "on every river row,
    max_future equals current") live in notebooks/validation.ipynb,
    since they need the full 500k-row dataframe.
    """

    # ---- Test 1: River rows reduce to the evaluator ----
    # On the river, no cards left to come, so max_future must equal the
    # best 5-card hand we can form right now.
    river_cases = [
        # (holding, flop, turn, river, expected_category)
        ('AhKh', 'QhJhTh', '2c', '3d', 'royal_flush'),
        ('AcAs', 'AhAd2c', '5h', '7d', 'four_of_a_kind'),
        ('KhKd', 'KsKc2h', '3d', '4c', 'four_of_a_kind'),
        ('9h9d', '9c2h3d', '4s', '5c', 'three_of_a_kind'),
        ('Ah2d', '7c8h9s', 'Tc', 'Jd', 'straight'),
        ('2c3d', '4h5s6c', '8d', 'Th', 'straight'),
    ]
    for holding, flop, turn, river, expected in river_cases:
        result = max_future_strength_fast(holding, flop, turn, river, 'River')
        assert result[0] == HAND_RANKS[expected], \
            f"River {holding}|{flop}{turn}{river}: " \
            f"expected {expected}, got category {result[0]}"
    print(f"✓ All {len(river_cases)} river-equivalence tests passed")

    # ---- Test 2: Flop draws are recognized correctly ----
    # On the flop, two cards are still to come.
    flop_cases = [
        # 4-to-the-royal: needs one card, two left to come
        ('AhKh', 'QhJh2c', '', '', 'royal_flush'),
        # Already have trips on the flop -> can reach quads with one more
        ('AhAd', 'AcKh2s', '', '', 'four_of_a_kind'),
        # Open-ended straight draw: 7c8h on 9s Ts 2c -> can reach straight
        ('7c8h', '9sTs2c', '', '', 'straight'),
        # Pocket pair on dry flop -> can reach quads with two more (set + 1)
        ('5h5d', '2c7sJh', '', '', 'four_of_a_kind'),
    ]
    for holding, flop, turn, river, expected in flop_cases:
        result = max_future_strength_fast(holding, flop, turn, river, 'Flop')
        assert result[0] == HAND_RANKS[expected], \
            f"Flop {holding}|{flop}: " \
            f"expected {expected}, got category {result[0]}"
    print(f"✓ All {len(flop_cases)} flop-draw tests passed")

    # ---- Test 3: max_future never decreases as we move down streets ----
    # Holding a hand fixed across streets, the max-future category should
    # be monotone non-increasing as more board cards are revealed (later
    # streets have fewer remaining draws).
    monotone_cases = [
        ('AhKh', 'QhJh2c', 'Ts', '7d'),  # Royal possible on flop, dies by river
        ('AhAd', 'AcKh2s', '7d', '9c'),  # Trips -> quads possible early
    ]
    for holding, flop, turn, river in monotone_cases:
        flop_cat = max_future_strength_fast(holding, flop, turn, river, 'Flop')[0]
        turn_cat = max_future_strength_fast(holding, flop, turn, river, 'Turn')[0]
        river_cat = max_future_strength_fast(holding, flop, turn, river, 'River')[0]
        assert flop_cat >= turn_cat >= river_cat, \
            f"Non-monotone categories on {holding}|{flop}{turn}{river}: " \
            f"flop={flop_cat}, turn={turn_cat}, river={river_cat}"
    print(f"✓ All {len(monotone_cases)} monotonicity tests passed")


if __name__ == "__main__":
    _test_max_future_strength()
