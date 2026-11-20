from __future__ import annotations

import pandas as pd

STREAK_NAMES = ["win", "loss", "no_score", "no_concede",
                "max_no_score", "max_no_concede"]


def add_streak_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Streak atual e recorde historico sem gol/sem sofrer gol, pre-jogo."""
    frame = frame.sort_values(["team_id", "date", "fixture_id"]).copy()
    outputs = []

    for _, g in frame.groupby("team_id", sort=False):
        g = g.copy()
        win = loss = no_score = no_concede = 0
        max_no_score = max_no_concede = 0
        current = {k: [] for k in STREAK_NAMES}

        for row in g.itertuples(index=False):
            current["win"].append(win)
            current["loss"].append(loss)
            current["no_score"].append(no_score)
            current["no_concede"].append(no_concede)
            current["max_no_score"].append(max_no_score)
            current["max_no_concede"].append(max_no_concede)

            gf, ga, pts = row.goals_for, row.goals_against, row.points
            win = win + 1 if pts == 3 else 0
            loss = loss + 1 if pts == 0 else 0
            no_score = no_score + 1 if gf == 0 else 0
            no_concede = no_concede + 1 if ga == 0 else 0
            max_no_score = max(max_no_score, no_score)
            max_no_concede = max(max_no_concede, no_concede)

        for key, values in current.items():
            g[f"{prefix}_{key}"] = values
        outputs.append(g)

    return pd.concat(outputs, ignore_index=True) if outputs else frame