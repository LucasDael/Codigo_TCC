from __future__ import annotations

import numpy as np
import pandas as pd

from competition import add_competition_features
from config import FINISHED_STATUSES, OUTPUT_PATH
from io_utils import read_input_csv, safe_numeric
from points_context import add_points_context_features
from poisson import poisson_probabilities_vectorized
from season_rank import add_last_season_rank
from team_table import build_team_match_table
from venue import add_venue_and_streak_history

LEAKAGE_COLS = [
    "goals_home", "goals_away", "teams_home_winner", "teams_away_winner",
    "score_halftime_home", "score_halftime_away",
    "score_fulltime_home", "score_fulltime_away",
    "score_extratime_home", "score_extratime_away",
    "score_penalty_home", "score_penalty_away",
]


def _prepare_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    f = fixtures.copy()
    f["date"] = pd.to_datetime(f["fixture_date"], utc=True, errors="coerce")
    safe_numeric(f, ["goals_home", "goals_away", "league_id", "league_season"])

    if "fixture_status_short" in f.columns:
        f = f[f["fixture_status_short"].isin(FINISHED_STATUSES)].copy()
    else:
        print("Aviso: coluna 'fixture_status_short' ausente; usando todos os jogos.")

    f = f.dropna(subset=["goals_home", "goals_away", "date"])
    f = f.drop_duplicates("fixture_id")
    f["target"] = np.select(
        [f["goals_home"] > f["goals_away"],
         f["goals_home"] == f["goals_away"]],
        [1, 0], default=2,
    ).astype(int)
    return f


def _build_side_features(team_matches: pd.DataFrame,
                         team_features: pd.DataFrame,
                         side: str) -> pd.DataFrame:
    """Extrai features da perspectiva de um lado (home/away)."""
    if side == "home":
        mask = team_matches["is_home"] == 1
        context_prefix = "home_"
    else:
        mask = team_matches["is_home"] == 0
        context_prefix = "away_"

    cols = [c for c in team_features.columns
            if c.startswith("all_") or c.startswith(context_prefix)]
    ids = team_matches.loc[mask, ["fixture_id", "team_id"]]
    merged = ids.merge(
        team_features[["fixture_id", "team_id"] + cols],
        on=["fixture_id", "team_id"], how="left",
    ).drop(columns=["team_id"])
    return merged.rename(columns={
        c: f"{side}_{c}" for c in merged.columns if c != "fixture_id"
    })


def _add_strength_features(result: pd.DataFrame) -> pd.DataFrame:
    """Forca ofensiva/defensiva relativa a media da competicao + Poisson."""
    for side in ["home", "away"]:
        gf = f"{side}_all_goals_for_mean_all"
        ga = f"{side}_all_goals_against_mean_all"
        if side == "home":
            league_gf = "competition_league_home_goals_mean"
            league_ga = "competition_league_away_goals_mean"
        else:
            league_gf = "competition_league_away_goals_mean"
            league_ga = "competition_league_home_goals_mean"

        if gf in result and league_gf in result:
            result[f"{side}_attack_strength"] = (
                result[gf] / result[league_gf].replace(0, np.nan)
            )
        if ga in result and league_ga in result:
            result[f"{side}_defense_concession_strength"] = (
                result[ga] / result[league_ga].replace(0, np.nan)
            )

    required = {
        "home_attack_strength", "away_attack_strength",
        "home_defense_concession_strength", "away_defense_concession_strength",
        "competition_league_home_goals_mean", "competition_league_away_goals_mean",
    }
    if required.issubset(result.columns):
        result["lambda_home"] = (
            result["competition_league_home_goals_mean"]
            * result["home_attack_strength"]
            * result["away_defense_concession_strength"]
        )
        result["lambda_away"] = (
            result["competition_league_away_goals_mean"]
            * result["away_attack_strength"]
            * result["home_defense_concession_strength"]
        )
        probs = poisson_probabilities_vectorized(
            result["lambda_home"], result["lambda_away"]
        )
        result = pd.concat([result, probs], axis=1)
    return result


def main() -> None:
    fixtures = read_input_csv("fixture_stats_clean.csv")
    team_stats = read_input_csv("fixture_stats_teams_clean.csv")

    team_matches = build_team_match_table(fixtures, team_stats)
    team_matches = add_points_context_features(team_matches)
    team_features = add_venue_and_streak_history(team_matches)

    f = _prepare_fixtures(fixtures)

    home_features = _build_side_features(team_matches, team_features, "home")
    away_features = _build_side_features(team_matches, team_features, "away")

    result = f[["fixture_id", "date", "league_id", "league_season",
                "teams_home_id", "teams_away_id", "target"]].copy()
    result = result.merge(home_features, on="fixture_id", how="left")
    result = result.merge(away_features, on="fixture_id", how="left")
    result = result.merge(add_competition_features(f), on="fixture_id", how="left")
    result = add_last_season_rank(result, f)
    result = _add_strength_features(result)

    result = result.drop(columns=[c for c in LEAKAGE_COLS if c in result.columns])
    result = result.sort_values(["date", "fixture_id"]).reset_index(drop=True)

    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Arquivo salvo: {OUTPUT_PATH.resolve()}")
    print(f"Partidas: {len(result):,}")
    print(f"Colunas: {len(result.columns):,}")
    print("Distribuicao target:")
    print(result["target"].value_counts().sort_index())


if __name__ == "__main__":
    main()