from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    GOALS_CAP, RED_CARDS_CAP, SHOTS_ON_TARGET_PER_GOAL_CAP,
    SHOTS_PER_GOAL_CAP, YELLOW_CARDS_CAP,
)

HISTORY_COLS = [
    "points_pct_match", "goals_for_norm", "goals_against_norm",
    "yellow_cards_norm", "red_cards_norm", "shots_per_goal_norm",
    "shots_on_per_goal_norm", "ball_possession",
    "goals_for", "goals_against", "points", "total_shots",
    "shots_on_goal", "yellow_cards", "fouls",
]


def clipped_ratio(numerator: pd.Series, denominator: pd.Series,
                  cap: float) -> pd.Series:
    """Razao acumulada, limitada ao teto; sem gols, assume o teto."""
    ratio = numerator / denominator.replace(0, np.nan)
    return ratio.fillna(cap).clip(lower=0, upper=cap) / cap


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["team_id", "date", "fixture_id"]).copy()
    frame["points_pct_match"] = frame["points"] / 3.0
    frame["goals_for_norm"] = frame["goals_for"].clip(0, GOALS_CAP) / GOALS_CAP
    frame["goals_against_norm"] = frame["goals_against"].clip(0, GOALS_CAP) / GOALS_CAP
    frame["yellow_cards_norm"] = frame["yellow_cards"].clip(0, YELLOW_CARDS_CAP) / YELLOW_CARDS_CAP
    frame["red_cards_norm"] = frame["red_cards"].clip(0, RED_CARDS_CAP) / RED_CARDS_CAP
    frame["shots_per_goal_norm"] = clipped_ratio(
        frame["total_shots"], frame["goals_for"], SHOTS_PER_GOAL_CAP
    )
    frame["shots_on_per_goal_norm"] = clipped_ratio(
        frame["shots_on_goal"], frame["goals_for"], SHOTS_ON_TARGET_PER_GOAL_CAP
    )
    return frame


def add_group_rolling_features(
    frame: pd.DataFrame,
    prefix: str,
    windows: list[int],
    ema_spans: list[int],
) -> pd.DataFrame:
    """Cria features pre-jogo por equipe, ordenadas cronologicamente."""
    frame = _prepare_frame(frame)
    feature_data: dict[str, pd.Series] = {}
    team_groups = frame["team_id"]

    for col in HISTORY_COLS:
        prior = frame.groupby("team_id", sort=False)[col].shift(1)
        feature_data[f"{prefix}_{col}_previous"] = prior
        grouped_prior = prior.groupby(team_groups, sort=False)

        for w in windows:
            feature_data[f"{prefix}_{col}_mean_{w}"] = (
                grouped_prior.rolling(w, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
            feature_data[f"{prefix}_{col}_std_{w}"] = (
                grouped_prior.rolling(w, min_periods=2).std()
                .reset_index(level=0, drop=True)
            )
        feature_data[f"{prefix}_{col}_mean_all"] = (
            grouped_prior.expanding(min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        feature_data[f"{prefix}_{col}_std_all"] = (
            grouped_prior.expanding(min_periods=2).std()
            .reset_index(level=0, drop=True)
        )
        for span in ema_spans:
            feature_data[f"{prefix}_{col}_ema_{span}"] = (
                grouped_prior.ewm(span=span, adjust=False, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )

    return pd.concat([frame, pd.DataFrame(feature_data, index=frame.index)], axis=1)