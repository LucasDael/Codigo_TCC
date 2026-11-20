from __future__ import annotations

import math

import numpy as np
import pandas as pd


def poisson_probabilities(lambda_home: float, lambda_away: float,
                          max_goals: int = 10) -> tuple[float, float, float]:
    """Probabilidades 1/X/2 via Poisson independente para gols das equipes."""
    if not np.isfinite(lambda_home) or not np.isfinite(lambda_away):
        return (np.nan, np.nan, np.nan)
    lambda_home = max(0.01, float(lambda_home))
    lambda_away = max(0.01, float(lambda_away))

    def pmf(k: int, lam: float) -> float:
        return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

    p_home = p_draw = p_away = 0.0
    for h in range(max_goals + 1):
        ph = pmf(h, lambda_home)
        for a in range(max_goals + 1):
            p = ph * pmf(a, lambda_away)
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def poisson_probabilities_vectorized(lambda_home: pd.Series,
                                     lambda_away: pd.Series,
                                     max_goals: int = 10) -> pd.DataFrame:
    """Versao vetorizada das probabilidades 1/X/2 para toda a base."""
    lh = pd.to_numeric(lambda_home, errors="coerce").clip(lower=0.01).to_numpy()
    la = pd.to_numeric(lambda_away, errors="coerce").clip(lower=0.01).to_numpy()
    valid = np.isfinite(lh) & np.isfinite(la)
    lh = np.where(valid, lh, 1.0)
    la = np.where(valid, la, 1.0)

    n = len(lh)
    ph = np.empty((n, max_goals + 1), dtype=float)
    pa = np.empty((n, max_goals + 1), dtype=float)
    ph[:, 0] = np.exp(-lh)
    pa[:, 0] = np.exp(-la)
    for k in range(1, max_goals + 1):
        ph[:, k] = ph[:, k - 1] * lh / k
        pa[:, k] = pa[:, k - 1] * la / k

    p_home = np.zeros(n)
    p_draw = np.zeros(n)
    p_away = np.zeros(n)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = ph[:, h] * pa[:, a]
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p

    total = p_home + p_draw + p_away
    out = pd.DataFrame({
        "poisson_prob_home_win": p_home / total,
        "poisson_prob_draw": p_draw / total,
        "poisson_prob_away_win": p_away / total,
    })
    out.loc[~valid, :] = np.nan
    out.index = lambda_home.index
    return out