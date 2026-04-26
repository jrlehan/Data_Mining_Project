"""Parsers for the betting-action strings in the dataset.

Two flavors:

`parse_preflop` / `parse_postflop` extract a small handful of summary
features from the action strings -- enough to capture preflop pressure
(how many raises, how big) and the immediate postflop context (current
bets/raises, last bet size).

`action_sequence_features` extracts a richer per-street breakdown:
counts of each action type and bet sizing on flop / turn / river. This
is what the XGBoost "natural features" model uses.

Format reminder: postflop action strings interleave action tokens with
'dealcards/<rank><suit>/' separators between streets, e.g.
'OOP_CHECK/IP_BET_5.0/dealcards/Td/OOP_CALL'.
"""

import re

# Pre-compiled regex (called millions of times, so the compile cost matters)
PREFLOP_BB_RE = re.compile(r'^([\d.]+)bb$')
POSTFLOP_BETRAISE_RE = re.compile(r'^(?:OOP|IP)_(?:BET|RAISE)_([\d.]+)$')
DEALCARDS_RE = re.compile(r'dealcards/[2-9TJQKA][hdcs]/')
POSTFLOP_TOKEN_RE = re.compile(
    r'^(OOP|IP)_(CHECK|CALL|BET|RAISE|FOLD)(?:_([\d.]+))?$'
)


def parse_preflop(action_str):
    """Extract num_raises, num_players, last_raise_bb from a preflop action string."""
    if not isinstance(action_str, str) or action_str == '':
        return {'num_raises': 0, 'num_players': 0, 'last_raise_bb': 0.0}

    tokens = action_str.split('/')
    actions = tokens[1::2]
    positions = set(tokens[0::2])

    raise_sizes = []
    for a in actions:
        m = PREFLOP_BB_RE.match(a)
        if m:
            raise_sizes.append(float(m.group(1)))

    return {
        'num_raises': len(raise_sizes),
        'num_players': len(positions),
        'last_raise_bb': raise_sizes[-1] if raise_sizes else 0.0,
    }


def parse_postflop(action_str):
    """Extract a summary of the *current* postflop street's betting.

    Specifically counts bets/raises and the last bet size on whichever
    street the player is currently being asked to act on (i.e. the
    final segment after the most recent dealcards/.../ marker).
    """
    if not isinstance(action_str, str) or action_str == '':
        return {'current_bets_raises': 0, 'last_bet_size': 0.0}

    segments = DEALCARDS_RE.split(action_str)
    current_street = segments[-1]
    tokens = current_street.split('/') if current_street else []

    bet_raise_sizes = []
    for t in tokens:
        m = POSTFLOP_BETRAISE_RE.match(t)
        if m:
            bet_raise_sizes.append(float(m.group(1)))

    return {
        'current_bets_raises': len(bet_raise_sizes),
        'last_bet_size': bet_raise_sizes[-1] if bet_raise_sizes else 0.0,
    }


def action_sequence_features(action_str):
    """Per-street breakdown of action counts and sizing.

    Returns a flat dict with keys like `flop_n_bet`, `flop_max_size`,
    `turn_n_check`, `river_sum_size`, etc. Empty / non-string input
    yields an empty dict (the caller fills missing keys with zero).
    """
    if not isinstance(action_str, str) or action_str == '':
        return {}

    parts = DEALCARDS_RE.split(action_str)
    street_names = ['flop', 'turn', 'river']

    feats = {}
    for i, seg in enumerate(parts):
        name = street_names[i] if i < 3 else f'extra{i}'
        tokens = seg.split('/') if seg else []

        n_bet = n_raise = n_check = n_call = 0
        sizes = []
        for t in tokens:
            m = POSTFLOP_TOKEN_RE.match(t)
            if not m:
                continue
            action = m.group(2)
            size = m.group(3)
            if action == 'BET':
                n_bet += 1
                if size:
                    sizes.append(float(size))
            elif action == 'RAISE':
                n_raise += 1
                if size:
                    sizes.append(float(size))
            elif action == 'CHECK':
                n_check += 1
            elif action == 'CALL':
                n_call += 1

        feats[f'{name}_n_bet'] = n_bet
        feats[f'{name}_n_raise'] = n_raise
        feats[f'{name}_n_check'] = n_check
        feats[f'{name}_n_call'] = n_call
        feats[f'{name}_max_size'] = max(sizes) if sizes else 0.0
        feats[f'{name}_sum_size'] = sum(sizes)
    return feats
