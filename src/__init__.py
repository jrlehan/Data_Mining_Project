"""Feature-engineering modules for the poker decision project.

Each module is self-contained and importable from a notebook:

    from src.hand_evaluator import best_hand_strength
    from src.max_strength import max_future_strength_fast
    from src.nut_strength import nut_strength
    from src.action_parser import parse_preflop, parse_postflop, action_sequence_features
    from src.features import (
        simplify_action, simplify_available_moves, parse_holding,
        POSITION_VALUES, STREET_VALUES,
        build_card_onehots, board_aggregate_features,
    )
    from src.cards import HAND_RANKS, CATEGORY_NAMES
"""
