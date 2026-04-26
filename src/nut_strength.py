"""Best hand any *opponent* could plausibly hold by the river.

`nut_strength` answers the dual of `max_future_strength`: given the
player's hole cards (which are blockers -- the opponent can't hold
those) and the visible board, what is the strongest 5-card hand any
opponent could end up with?

Same trick as `max_strength`: walk the hand-rank ladder top-down and
stop at the first reachable category. The differences are
  * the opponent gets exactly 2 hole cards from the unblocked deck, and
  * future board cards (if we're not on the river) are shared with us,
    so we count them toward both reachability and uniqueness.
"""

from collections import Counter

from .cards import ALL_SUITS, FULL_DECK, HAND_RANKS, RANK_VALUES, cards_to_strings


def nut_strength(holding, flop, turn, river, street):
    """Best 5-card hand category any opponent could still reach."""
    my_hole = cards_to_strings(holding)
    board_known = cards_to_strings(flop)
    if street == 'Turn':
        board_known += cards_to_strings(turn)
        future_board = 1
    elif street == 'River':
        board_known += cards_to_strings(turn) + cards_to_strings(river)
        future_board = 0
    else:  # Flop
        future_board = 2

    # Opponent can't hold anything we hold or that's already on the board.
    blocked = set(my_hole + board_known)
    available = [c for c in FULL_DECK if c not in blocked]

    # Pre-index available cards by rank and by (suit, rank) for fast lookups.
    avail_by_rank = Counter(RANK_VALUES[c[0]] for c in available)
    avail_by_suit_rank = {
        s: {RANK_VALUES[c[0]] for c in available if c[1] == s}
        for s in ALL_SUITS
    }

    board_parsed = [(RANK_VALUES[c[0]], c[1]) for c in board_known]
    board_rank_counts = Counter(r for r, _ in board_parsed)
    board_rank_set = set(board_rank_counts)
    board_by_suit = {
        s: {r for r, ss in board_parsed if ss == s}
        for s in ALL_SUITS
    }

    total_slots = 2 + future_board  # opp's 2 holes + remaining board

    def can_get_ranks(needed_ranks):
        """Can opp + future board collectively produce every rank in `needed_ranks`?"""
        missing = needed_ranks - board_rank_set
        if len(missing) > total_slots:
            return False
        for r in missing:
            if avail_by_rank.get(r, 0) == 0:
                return False
        return True

    def can_get_suited(needed_suit, needed_ranks):
        """Can opp + future board produce 5 of `needed_suit` with these ranks?"""
        have = board_by_suit[needed_suit]
        missing = needed_ranks - have
        if len(missing) > total_slots:
            return False
        for r in missing:
            if r not in avail_by_suit_rank[needed_suit]:
                return False
        return True

    # ---- Royal flush ----
    for suit in ALL_SUITS:
        if can_get_suited(suit, {14, 13, 12, 11, 10}):
            return (HAND_RANKS['royal_flush'],)

    # ---- Straight flush ----
    best_sf_high = 0
    for suit in ALL_SUITS:
        for high in range(14, 5, -1):
            if can_get_suited(suit, set(range(high - 4, high + 1))):
                best_sf_high = max(best_sf_high, high)
                break
        if can_get_suited(suit, {14, 2, 3, 4, 5}) and best_sf_high < 5:
            best_sf_high = max(best_sf_high, 5)
    if best_sf_high > 0:
        return (HAND_RANKS['straight_flush'], best_sf_high)

    # ---- Four of a kind ----
    for r in range(14, 1, -1):
        board_have = board_rank_counts.get(r, 0)
        need = 4 - board_have
        if need <= total_slots and avail_by_rank.get(r, 0) >= need:
            kicker = 14 if avail_by_rank.get(14, 0) > 0 else max(
                (rr for rr in range(14, 1, -1)
                 if rr != r and avail_by_rank.get(rr, 0) > 0),
                default=2,
            )
            return (HAND_RANKS['four_of_a_kind'], r, kicker)

    # ---- Full house ----
    best_fh = None
    for trip_r in range(14, 1, -1):
        trip_need = max(0, 3 - board_rank_counts.get(trip_r, 0))
        if trip_need > total_slots or avail_by_rank.get(trip_r, 0) < trip_need:
            continue
        for pair_r in range(14, 1, -1):
            if pair_r == trip_r:
                continue
            pair_need = max(0, 2 - board_rank_counts.get(pair_r, 0))
            if (trip_need + pair_need <= total_slots
                    and avail_by_rank.get(pair_r, 0) >= pair_need):
                best_fh = (HAND_RANKS['full_house'], trip_r, pair_r)
                break
        if best_fh:
            break

    # ---- Flush ----
    best_flush = None
    for suit in ALL_SUITS:
        have = sorted(board_by_suit[suit], reverse=True)
        need = max(0, 5 - len(have))
        if need <= total_slots:
            pool = sorted(avail_by_suit_rank[suit], reverse=True)
            if len(pool) >= need:
                top_fill = pool[:need]
                flush_ranks = sorted(have + top_fill, reverse=True)[:5]
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
    best_straight_high = 0
    for high in range(14, 5, -1):
        if can_get_ranks(set(range(high - 4, high + 1))):
            best_straight_high = high
            break
    if best_straight_high == 0 and can_get_ranks({14, 2, 3, 4, 5}):
        best_straight_high = 5
    if best_straight_high > 0:
        return (HAND_RANKS['straight'], best_straight_high)

    # ---- Three of a kind ----
    for r in range(14, 1, -1):
        need = max(0, 3 - board_rank_counts.get(r, 0))
        if need <= total_slots and avail_by_rank.get(r, 0) >= need:
            return (HAND_RANKS['three_of_a_kind'], r, 14, 13)

    # ---- Two pair ----
    for high_r in range(14, 1, -1):
        hn = max(0, 2 - board_rank_counts.get(high_r, 0))
        if hn > total_slots or avail_by_rank.get(high_r, 0) < hn:
            continue
        for low_r in range(high_r - 1, 1, -1):
            ln = max(0, 2 - board_rank_counts.get(low_r, 0))
            if hn + ln <= total_slots and avail_by_rank.get(low_r, 0) >= ln:
                return (HAND_RANKS['two_pair'], high_r, low_r, 14)

    # ---- Pair ----
    for r in range(14, 1, -1):
        need = max(0, 2 - board_rank_counts.get(r, 0))
        if need <= total_slots and avail_by_rank.get(r, 0) >= need:
            return (HAND_RANKS['pair'], r, 14, 13, 12)

    return (HAND_RANKS['high_card'], 14, 13, 12, 11, 10)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_nut_strength():
    """Sanity tests for the nut routine.

    Run via `python -m src.nut_strength`. Like `max_strength`, these are
    data-independent. Data-dependent invariants (e.g. blocker-effect
    distributions across the full 500k rows) live in
    notebooks/validation.ipynb.
    """

    # ---- Test 1: Paired board on the river -> opponent can hit quads ----
    # Board 5h5d2c7s9c on the river; we hold AcKd. Two 5s remain in the
    # deck (5c, 5s), and opp has 2 hole cards -> opp can hold 5c5s for
    # quad fives.
    result = nut_strength('AcKd', '5h5d2c', '7s', '9c', 'River')
    assert result[0] == HAND_RANKS['four_of_a_kind'], \
        f"Paired board river: expected quads, got {result}"
    assert result[1] == 5, f"Paired board river: expected quad fives, got {result}"

    # ---- Test 2: Monotone flop with royal possible ----
    # AhKhQh flop, we hold AcKc. Opp could hold Jh+Th for a royal flush.
    result = nut_strength('AcKc', 'AhKhQh', '', '', 'Flop')
    assert result[0] == HAND_RANKS['royal_flush'], \
        f"Monotone flop with royal possible: expected royal, got {result}"

    # ---- Test 3: We block the royal ----
    # QhJhTh flop, we hold AhKh. Opp can't make royal (we hold Ah, Kh)
    # but can still make a Q-high straight flush with 9h+8h (board has
    # Qh+Jh+Th, opp adds 9h+8h -> 8-9-T-J-Q SF, which is 12-high).
    result = nut_strength('AhKh', 'QhJhTh', '', '', 'Flop')
    assert result[0] == HAND_RANKS['straight_flush'], \
        f"Royal-blocked flop: expected SF, got {result}"
    assert result[1] == 12, \
        f"Royal-blocked flop: expected 12-high SF, got high={result[1]}"

    # ---- Test 4: Even a dry rainbow flop allows a SF given 2 future cards ----
    # 2h7d9c flop, we hold AcKd. Surprising but true: opp has 2 hole
    # cards + 2 future board cards = 4 unknowns; opp holds e.g. Tc+Jc
    # and the future board brings Qc+Kc, joining with the board's 9c
    # to make a 9-T-J-Q-K straight flush.
    result = nut_strength('AcKd', '2h7d9c', '', '', 'Flop')
    assert result[0] == HAND_RANKS['straight_flush'], \
        f"Dry rainbow flop: expected SF (reachable via future cards), got {result}"

    # ---- Test 5: River with no draws left ----
    # Board 2h7d9cKsAd, we hold JcTh. Opp gets 2 hole cards from the
    # remaining deck. Best opp can do is hit trips by holding a pocket
    # pair matching the board (e.g. AA -> AAA with A-K-9 kickers).
    # Quads aren't possible (only 1 of each board rank left, opp has 2 cards).
    result = nut_strength('JcTh', '2h7d9c', 'Ks', 'Ad', 'River')
    assert result[0] == HAND_RANKS['three_of_a_kind'], \
        f"Dry river: expected trips, got {result}"
    assert result[1] == 14, \
        f"Dry river: expected trip aces (highest possible), got rank={result[1]}"

    print("✓ All 5 nut-strength tests passed")


if __name__ == "__main__":
    _test_nut_strength()
