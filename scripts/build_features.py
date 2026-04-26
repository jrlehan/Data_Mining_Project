"""Build all preprocessed dataframes for the levels-of-poker analysis.

Run this once before opening main_notebook.ipynb. It produces eight
parquet files under data/processed/, one per stage of the analysis:

    df_simplified.parquet           # cleaned + decision_category
    df_own_hand.parquet             # Level 1 features
    df_hand_strength_final.parquet  # Level 2 (hand strength added)
    df_max_strength_final.parquet   # Level 3 (max-future added)
    df_nut_strength_final.parquet   # Level 4 (nut added)
    df_context.parquet              # Level 5 (position + pot added)
    df_action.parquet               # Level 6 (action sequences added)
    df_rich.parquet                 # XGBoost natural features

Usage:
    python -m scripts.build_features <path-to-raw-csv>

Or from a notebook:
    !python -m scripts.build_features data/raw/postflop_500k_train_set_game_scenario_information.csv

The script is idempotent: if data/processed/<name>.parquet already
exists and is newer than its inputs, that stage is skipped. Pass
--force to rebuild everything.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.cards import CATEGORY_NAMES
from src.hand_evaluator import best_hand_strength
from src.max_strength import max_future_strength_fast
from src.nut_strength import nut_strength
from src.action_parser import (
    parse_preflop, parse_postflop, action_sequence_features,
)
from src.features import (
    simplify_action, simplify_available_moves, parse_holding,
    POSITION_VALUES, STREET_VALUES,
    build_card_onehots, board_aggregate_features,
)


PROCESSED_DIR = "data/processed"


def _save(df, name):
    """Save a dataframe to parquet, dropping any tuple-typed columns first.

    Parquet doesn't support Python tuple objects, but we don't need to
    persist the raw `*_tuple` columns -- the ordinal `*_rank` columns
    carry the same information for downstream models.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    tuple_cols = [c for c in df.columns if c.endswith('_tuple')]
    out = df.drop(columns=tuple_cols) if tuple_cols else df
    path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
    out.to_parquet(path, index=True)
    print(f"  saved {path}  ({len(out)} rows × {out.shape[1]} cols)")


# ---------------------------------------------------------------------------
# Stage 1: load + clean + simplify
# ---------------------------------------------------------------------------

def build_simplified(raw_csv_path):
    print("[1/8] Loading and cleaning raw CSV...")
    df = pd.read_csv(raw_csv_path)
    df = df.rename(columns={"Unnamed: 0": "Index"})
    df.columns = df.columns.str.strip()
    df["pot_size"] = pd.to_numeric(df["pot_size"], errors="coerce")

    cat_cols = [
        "preflop_action", "board_flop", "board_turn", "board_river",
        "aggressor_position", "postflop_action", "evaluation_at",
        "hero_position", "holding", "correct_decision",
    ]
    for col in cat_cols:
        df[col] = df[col].astype(str)

    df["available_moves_simplified"] = df["available_moves"].apply(
        simplify_available_moves
    )
    df["decision_category"] = df["correct_decision"].apply(simplify_action)
    df_simplified = df.drop(columns=["correct_decision", "available_moves", "Index"])

    # Parquet can't store list-typed cells; flatten available_moves_simplified
    # to a comma-joined string for storage. The notebook re-splits if needed.
    df_simplified["available_moves_simplified"] = (
        df_simplified["available_moves_simplified"].apply(",".join)
    )
    _save(df_simplified, "df_simplified")
    return df, df_simplified


# ---------------------------------------------------------------------------
# Stage 2: Level 1 own-hand features
# ---------------------------------------------------------------------------

def build_own_hand(df_simplified):
    print("[2/8] Building own-hand features...")
    df_own_hand = df_simplified.drop(columns=[
        "preflop_action", "board_flop", "board_turn", "board_river",
        "aggressor_position", "postflop_action", "evaluation_at",
        "pot_size", "hero_position",
    ]).copy()

    ranks = df_own_hand["holding"].apply(parse_holding)
    df_own_hand["rank1"] = ranks.apply(lambda r: r[0])
    df_own_hand["rank2"] = ranks.apply(lambda r: r[1])
    df_own_hand["is_pair"] = (
        df_own_hand["rank1"] == df_own_hand["rank2"]
    ).astype(int)
    df_own_hand["rank_sum"] = df_own_hand["rank1"] + df_own_hand["rank2"]
    df_own_hand["rank_max"] = df_own_hand[["rank1", "rank2"]].max(axis=1)

    moves = df_own_hand["available_moves_simplified"].str.split(",")
    df_own_hand["facing_bet"] = moves.apply(lambda m: int("Call" in m))
    df_own_hand["can_bet_raise"] = moves.apply(lambda m: int("Bet/Raise" in m))

    df_own_hand = df_own_hand.drop(columns=[
        "holding", "available_moves_simplified", "rank1", "rank2",
    ])
    _save(df_own_hand, "df_own_hand")
    return df_own_hand


# ---------------------------------------------------------------------------
# Stage 3: Level 2 hand strength
# ---------------------------------------------------------------------------

def build_hand_strength(df, df_own_hand):
    print("[3/8] Computing hand strength (this takes a few minutes)...")
    tqdm.pandas(desc="hand strength")

    # Working dataframe carrying the columns we need to compute strength
    df_hand_strength = df_own_hand.assign(
        holding=df["holding"],
        flop=df["board_flop"],
        turn=df["board_turn"],
        river=df["board_river"],
        street=df["evaluation_at"],
    )
    df_hand_strength["hand_strength_tuple"] = df_hand_strength.progress_apply(
        lambda row: best_hand_strength(
            row["holding"], row["flop"], row["turn"], row["river"], row["street"]
        ),
        axis=1,
    )
    df_hand_strength["hand_strength"] = (
        df_hand_strength["hand_strength_tuple"]
        .rank(method="dense")
        .astype("Int64")
    )

    # Build the "presentation" dataframe used by the Level 2 model
    keep_cols = [
        "decision_category", "is_pair", "rank_sum", "rank_max",
        "can_bet_raise", "facing_bet",
    ]
    df_hand_strength_final = df_hand_strength[keep_cols].copy()
    df_hand_strength_final["hand_category"] = (
        df_hand_strength["hand_strength_tuple"].apply(
            lambda t: CATEGORY_NAMES[t[0]] if t is not None else None
        )
    )
    df_hand_strength_final["hand_rank"] = df_hand_strength["hand_strength"]
    df_hand_strength_final["street"] = (
        df_hand_strength["street"].map(STREET_VALUES)
    )
    _save(df_hand_strength_final, "df_hand_strength_final")
    return df_hand_strength, df_hand_strength_final


# ---------------------------------------------------------------------------
# Stage 4: Level 3 max future strength
# ---------------------------------------------------------------------------

def build_max_strength(df_hand_strength, df_hand_strength_final):
    print("[4/8] Computing max future strength (this takes a few minutes)...")
    tqdm.pandas(desc="max future")

    df_max_strength = df_hand_strength.assign(
        max_future_tuple=df_hand_strength.progress_apply(
            lambda row: max_future_strength_fast(
                row["holding"], row["flop"], row["turn"], row["river"], row["street"]
            ),
            axis=1,
        )
    )
    # Co-rank current and future tuples on a shared scale
    all_tuples = pd.concat([
        df_max_strength["hand_strength_tuple"],
        df_max_strength["max_future_tuple"],
    ])
    unified = all_tuples.rank(method="dense").astype("Int64")
    n = len(df_max_strength)
    df_max_strength["hand_rank"] = unified.iloc[:n].values
    df_max_strength["max_future_rank"] = unified.iloc[n:].values

    df_max_strength_final = df_hand_strength_final.copy()
    df_max_strength_final["max_future_rank"] = df_max_strength["max_future_rank"]
    df_max_strength_final["hand_rank"] = df_max_strength["hand_rank"]
    _save(df_max_strength_final, "df_max_strength_final")
    return df_max_strength, df_max_strength_final


# ---------------------------------------------------------------------------
# Stage 5: Level 4 nut strength
# ---------------------------------------------------------------------------

def build_nut_strength(df_max_strength, df_max_strength_final):
    print("[5/8] Computing opponent nut strength (this takes a few minutes)...")
    tqdm.pandas(desc="nut strength")

    df_nut_strength = df_max_strength.assign(
        nut_tuple=df_max_strength.progress_apply(
            lambda row: nut_strength(
                row["holding"], row["flop"], row["turn"], row["river"], row["street"]
            ),
            axis=1,
        )
    )
    all_tuples = pd.concat([
        df_nut_strength["hand_strength_tuple"],
        df_nut_strength["max_future_tuple"],
        df_nut_strength["nut_tuple"],
    ])
    unified = all_tuples.rank(method="dense").astype("Int64")
    n = len(df_nut_strength)
    df_nut_strength["hand_rank"] = unified.iloc[:n].values
    df_nut_strength["max_future_rank"] = unified.iloc[n:2 * n].values
    df_nut_strength["opponent_nut_rank"] = unified.iloc[2 * n:].values

    df_nut_strength_final = df_max_strength_final.copy()
    df_nut_strength_final["opponent_nut_rank"] = df_nut_strength["opponent_nut_rank"]
    df_nut_strength_final["hand_rank"] = df_nut_strength["hand_rank"]
    df_nut_strength_final["max_future_rank"] = df_nut_strength["max_future_rank"]
    _save(df_nut_strength_final, "df_nut_strength_final")
    return df_nut_strength_final


# ---------------------------------------------------------------------------
# Stage 6: Level 5 basic context
# ---------------------------------------------------------------------------

def build_context(df, df_nut_strength_final):
    print("[6/8] Adding position and pot context...")
    df_context = df_nut_strength_final.copy()
    df_context["pot_size"] = df["pot_size"]
    df_context["hero_position"] = df["hero_position"].map(POSITION_VALUES)
    df_context["aggressor_position"] = df["aggressor_position"].map(POSITION_VALUES)
    _save(df_context, "df_context")
    return df_context


# ---------------------------------------------------------------------------
# Stage 7: Level 6 action sequences
# ---------------------------------------------------------------------------

def build_action(df, df_context):
    print("[7/8] Parsing action sequences...")
    df_action = df_context.copy()

    preflop_records = [
        parse_preflop(s) for s in tqdm(df["preflop_action"], desc="preflop")
    ]
    preflop_feats = pd.DataFrame.from_records(preflop_records, index=df.index)

    postflop_records = [
        parse_postflop(s) for s in tqdm(df["postflop_action"], desc="postflop")
    ]
    postflop_feats = pd.DataFrame.from_records(postflop_records, index=df.index)

    df_action["pre_num_raises"] = preflop_feats["num_raises"]
    df_action["pre_num_players"] = preflop_feats["num_players"]
    df_action["pre_last_raise_bb"] = preflop_feats["last_raise_bb"]
    df_action["post_bets_raises"] = postflop_feats["current_bets_raises"]
    df_action["post_last_bet_pct_pot"] = (
        postflop_feats["last_bet_size"] / df_action["pot_size"]
    ).fillna(0).replace([np.inf, -np.inf], 0)

    _save(df_action, "df_action")
    return df_action


# ---------------------------------------------------------------------------
# Stage 8: rich / natural features for XGBoost
# ---------------------------------------------------------------------------

def build_rich(df, df_action):
    print("[8/8] Building card one-hots, board aggregates, action features...")
    # We need the original card columns for these helpers, so reconstruct
    # a slim dataframe with the columns they expect.
    df_for_feats = pd.DataFrame({
        "holding": df["holding"],
        "board_flop": df["board_flop"],
        "board_turn": df["board_turn"],
        "board_river": df["board_river"],
        "evaluation_at": df["evaluation_at"],
    }, index=df.index)

    print("  card one-hots...")
    card_feats = build_card_onehots(df_for_feats)

    print("  board aggregates...")
    board_feats = board_aggregate_features(df_for_feats)

    print("  action sequences...")
    action_records = [
        action_sequence_features(s)
        for s in tqdm(df["postflop_action"], desc="actions")
    ]
    action_feats = pd.DataFrame.from_records(
        action_records, index=df.index
    ).fillna(0)

    df_rich = pd.concat([df_action, card_feats, board_feats, action_feats], axis=1)
    df_rich = df_rich.loc[:, ~df_rich.columns.duplicated()]
    _save(df_rich, "df_rich")
    return df_rich


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_csv",
        help="Path to postflop_500k_train_set_game_scenario_information.csv",
    )
    args = parser.parse_args()

    if not os.path.exists(args.raw_csv):
        print(f"ERROR: input file not found: {args.raw_csv}", file=sys.stderr)
        sys.exit(1)

    df, df_simplified = build_simplified(args.raw_csv)
    df_own_hand = build_own_hand(df_simplified)
    df_hand_strength, df_hand_strength_final = build_hand_strength(df, df_own_hand)
    df_max_strength, df_max_strength_final = build_max_strength(
        df_hand_strength, df_hand_strength_final
    )
    df_nut_strength_final = build_nut_strength(df_max_strength, df_max_strength_final)
    df_context = build_context(df, df_nut_strength_final)
    df_action = build_action(df, df_context)
    build_rich(df, df_action)

    print("\nAll preprocessed dataframes written to data/processed/.")


if __name__ == "__main__":
    main()
