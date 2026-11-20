from __future__ import annotations

import numpy as np
import pandas as pd

from io_utils import safe_numeric

COMPETITION_RAW_COLS = [
    "league_total_goals",
    "league_home_goals",
    "league_away_goals",
    "league_home_points_pct",
    "league_away_points_pct",
]


def add_competition_features(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Medias/desvios da competicao usando apenas partidas anteriores."""
    f = fixtures.copy()
    f["date"] = pd.to_datetime(f["fixture_date"], utc=True, errors="coerce")
    safe_numeric(f, ["goals_home", "goals_away"])
    f = f.sort_values(["league_id", "league_season", "date", "fixture_id"]).copy()

    has_goals = f["goals_home"].notna() & f["goals_away"].notna()

    f["league_total_goals"] = np.where(
        has_goals, f["goals_home"] + f["goals_away"], np.nan
    )
    f["league_home_goals"] = np.where(has_goals, f["goals_home"], np.nan)
    f["league_away_goals"] = np.where(has_goals, f["goals_away"], np.nan)

    home_pts = np.select(
        [f["goals_home"] > f["goals_away"], f["goals_home"] == f["goals_away"]],
        [1.0, 1 / 3], default=0.0,
    )
    away_pts = np.select(
        [f["goals_away"] > f["goals_home"], f["goals_home"] == f["goals_away"]],
        [1.0, 1 / 3], default=0.0,
    )
    f["league_home_points_pct"] = np.where(has_goals, home_pts, np.nan)
    f["league_away_points_pct"] = np.where(has_goals, away_pts, np.nan)

    group_keys = [f["league_id"], f["league_season"]]
    for col in COMPETITION_RAW_COLS:
        prior = f.groupby(["league_id", "league_season"], sort=False)[col].shift(1)
        grouped = prior.groupby(group_keys, sort=False)
        f[f"competition_{col}_mean"] = grouped.transform(
            lambda s: s.expanding(1).mean()
        )
        f[f"competition_{col}_std"] = grouped.transform(
            lambda s: s.expanding(2).std()
        )

    out_cols = ["fixture_id"] + [c for c in f.columns if c.startswith("competition_")]
    return f[out_cols]