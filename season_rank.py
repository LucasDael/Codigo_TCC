from __future__ import annotations

import pandas as pd

from io_utils import read_input_csv, safe_numeric

STANDINGS_FILE = "season_standings_clean.csv"


def add_last_season_rank(result: pd.DataFrame,
                         fixtures: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a colocacao da equipe na temporada anterior, sem usar a atual."""
    try:
        standings = read_input_csv(STANDINGS_FILE)
    except FileNotFoundError:
        print(f"Aviso: {STANDINGS_FILE} nao encontrado; rank anterior omitido.")
        return result

    required = {"season", "team_id", "rank"}
    if not required.issubset(standings.columns):
        print("Aviso: colunas de temporada/classificacao ausentes; rank anterior omitido.")
        return result

    standings = standings[["season", "team_id", "rank"]].copy()
    safe_numeric(standings, ["season", "team_id", "rank"])
    standings = standings.dropna(subset=["season", "team_id", "rank"])
    standings["season"] = standings["season"].astype(int)
    standings = standings.drop_duplicates(["season", "team_id"], keep="last")

    team_seasons = fixtures[
        ["fixture_id", "league_season", "teams_home_id", "teams_away_id"]
    ].copy()
    team_seasons["league_season"] = pd.to_numeric(
        team_seasons["league_season"], errors="coerce"
    )
    team_seasons["previous_season"] = team_seasons["league_season"] - 1

    home_pairs = team_seasons[["fixture_id", "previous_season", "teams_home_id"]].rename(
        columns={"teams_home_id": "team_id"}
    )
    away_pairs = team_seasons[["fixture_id", "previous_season", "teams_away_id"]].rename(
        columns={"teams_away_id": "team_id"}
    )

    lookup = pd.concat([home_pairs, away_pairs], ignore_index=True)
    lookup["team_id"] = pd.to_numeric(lookup["team_id"], errors="coerce")
    lookup["previous_season"] = pd.to_numeric(lookup["previous_season"], errors="coerce")

    lookup = lookup.merge(
        standings,
        left_on=["previous_season", "team_id"],
        right_on=["season", "team_id"],
        how="left",
    )[["fixture_id", "team_id", "rank"]].drop_duplicates(["fixture_id", "team_id"])

    home_rank = lookup.merge(
        home_pairs[["fixture_id", "team_id"]],
        on=["fixture_id", "team_id"], how="inner",
    ).rename(columns={"team_id": "teams_home_id", "rank": "home_last_season_rank"})

    away_rank = lookup.merge(
        away_pairs[["fixture_id", "team_id"]],
        on=["fixture_id", "team_id"], how="inner",
    ).rename(columns={"team_id": "teams_away_id", "rank": "away_last_season_rank"})

    result = result.merge(
        home_rank[["fixture_id", "home_last_season_rank"]],
        on="fixture_id", how="left",
    )
    result = result.merge(
        away_rank[["fixture_id", "away_last_season_rank"]],
        on="fixture_id", how="left",
    )
    return result