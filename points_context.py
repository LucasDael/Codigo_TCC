from __future__ import annotations

import numpy as np
import pandas as pd

RENAME = {
    "points_pct_all": "all_points_pct_all",
    "points_pct_home": "all_points_pct_home",
    "points_pct_away": "all_points_pct_away",
    "points_pct_advantage": "all_points_pct_advantage",
    "points_pct_disadvantage": "all_points_pct_disadvantage",
}


def _annotate_opponent_red_cards(x: pd.DataFrame) -> pd.DataFrame:
    """Calcula cartoes vermelhos do adversario a partir do total por fixture.

    Cada fixture aparece duas vezes (mandante e visitante), entao
    'total - proprios' devolve os cartoes do adversario.
    """
    fixture_total_red = x.groupby("fixture_id")["red_cards"].transform("sum")
    x["opponent_red_cards"] = fixture_total_red - x["red_cards"]
    return x


def add_points_context_features(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Percentual de pontos geral/casa/fora e pontos em vantagem/desvantagem."""
    x = team_matches.copy().sort_values(
        ["team_id", "date", "fixture_id"]
    ).reset_index(drop=True)
    x = _annotate_opponent_red_cards(x)

    result = []
    for _, g in x.groupby("team_id", sort=False):
        g = g.copy()

        prior_points = g["points"].shift(1)
        prior_masks = {
            "home":         g["is_home"].eq(1).shift(1, fill_value=False),
            "away":         g["is_home"].eq(0).shift(1, fill_value=False),
            "advantage":    g["opponent_red_cards"].gt(0).shift(1, fill_value=False),
            "disadvantage": g["red_cards"].gt(0).shift(1, fill_value=False),
        }

        for label, mask in prior_masks.items():
            pts = prior_points.where(mask)
            count = mask.astype(int)
            # Grupo ja e uma unica equipe -> cumsum direto basta.
            num = pts.fillna(0).cumsum()
            den = count.cumsum() * 3
            g[f"points_pct_{label}"] = (num / den.replace(0, np.nan)).clip(0, 1)

        g["points_pct_all"] = prior_points.expanding(1).mean() / 3.0
        g = g.rename(columns=RENAME)
        result.append(g)

    return pd.concat(result, ignore_index=True)