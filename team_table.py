from __future__ import annotations

import numpy as np
import pandas as pd

from config import FINISHED_STATUSES
from io_utils import safe_numeric

STAT_COLS = [
    "total_passes", "completed_passes", "pass_percentage", "assists",
    "expected_goals", "shots_on_goal", "shots_off_goal", "shots_insidebox",
    "shots_outsidebox", "total_shots", "blocked_shots", "fouls",
    "corner_kicks", "offsides", "ball_possession", "yellow_cards",
    "red_cards", "goalkeeper_saves", "goals_prevented",
]

REQUIRED_STAT_COLS = [
    "total_shots", "shots_on_goal", "ball_possession", "yellow_cards",
    "red_cards", "fouls",
]


def build_team_match_table(fixtures: pd.DataFrame,
                           team_stats: pd.DataFrame) -> pd.DataFrame:
    """Transforma cada partida em duas observacoes: uma por equipe."""
    f = fixtures.copy()
    f["date"] = pd.to_datetime(f["fixture_date"], utc=True, errors="coerce")
    safe_numeric(f, ["goals_home", "goals_away", "league_id", "league_season"])

    if "fixture_status_short" in f.columns:
        f = f[f["fixture_status_short"].isin(FINISHED_STATUSES)].copy()
    f = f.dropna(subset=["date", "goals_home", "goals_away",
                          "teams_home_id", "teams_away_id"])
    f = f.drop_duplicates("fixture_id")

    s = team_stats.copy()
    safe_numeric(s, STAT_COLS)
    s = s.drop_duplicates(["fixture_id", "team_id"], keep="last")
    keep = ["fixture_id", "team_id"] + [c for c in STAT_COLS if c in s.columns]
    s = s[keep]

    base_cols = ["fixture_id", "date", "league_id", "league_season",
                 "goals_home", "goals_away"]

    home = f[base_cols + ["teams_home_id", "teams_away_id"]].copy()
    home = home.rename(columns={
        "teams_home_id": "team_id", "teams_away_id": "opponent_id",
        "goals_home": "goals_for", "goals_away": "goals_against",
    })
    home["is_home"] = 1

    away = f[base_cols + ["teams_away_id", "teams_home_id"]].copy()
    away = away.rename(columns={
        "teams_away_id": "team_id", "teams_home_id": "opponent_id",
        "goals_away": "goals_for", "goals_home": "goals_against",
    })
    away["is_home"] = 0

    teams = pd.concat([home, away], ignore_index=True)
    teams = teams.merge(s, on=["fixture_id", "team_id"], how="left")

    teams["points"] = np.select(
        [teams["goals_for"] > teams["goals_against"],
         teams["goals_for"] == teams["goals_against"]],
        [3, 1], default=0,
    )

    for col in REQUIRED_STAT_COLS:
        if col not in teams.columns:
            teams[col] = np.nan
    teams["yellow_cards"] = teams["yellow_cards"].fillna(0)
    teams["red_cards"] = teams["red_cards"].fillna(0)

    return teams.sort_values(["team_id", "date", "fixture_id"]).reset_index(drop=True)