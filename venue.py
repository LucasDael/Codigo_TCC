from __future__ import annotations

import pandas as pd

from config import EMA_SPANS, STREAK_CAP, WINDOWS
from rolling import add_group_rolling_features
from streaks import add_streak_features

STREAK_SUFFIXES = ["win", "loss", "no_score", "no_concede",
                   "max_no_score", "max_no_concede"]


def _feature_subset(df: pd.DataFrame, context: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(context + "_")]
    return df[["fixture_id", "team_id"] + cols]


def add_venue_and_streak_history(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Calcula historico geral e historico condicionado a mando de campo."""
    all_hist = add_streak_features(
        add_group_rolling_features(team_matches, "all", WINDOWS, EMA_SPANS), "all"
    )
    home_hist = add_streak_features(
        add_group_rolling_features(
            team_matches[team_matches["is_home"] == 1], "home", WINDOWS, EMA_SPANS
        ), "home"
    )
    away_hist = add_streak_features(
        add_group_rolling_features(
            team_matches[team_matches["is_home"] == 0], "away", WINDOWS, EMA_SPANS
        ), "away"
    )

    out = _feature_subset(all_hist, "all")
    out = out.merge(_feature_subset(home_hist, "home"),
                    on=["fixture_id", "team_id"], how="left")
    out = out.merge(_feature_subset(away_hist, "away"),
                    on=["fixture_id", "team_id"], how="left")

    for context in ["all", "home", "away"]:
        for name in STREAK_SUFFIXES:
            col = f"{context}_{name}"
            if col in out.columns:
                out[f"{col}_norm"] = out[col].clip(0, STREAK_CAP) / STREAK_CAP
    return out