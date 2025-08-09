#!/usr/bin/env python
"""
Script to fetch raw data for the current season, merge with the previous season's raw data,
process matches and train Poisson models for all leagues.
"""
import os
import pandas as pd
from config.leagues import LEAGUES
from src.data.fetch import main as fetch_main, get_current_season
from src.data.process import process_matches
from src.models.train import train_league
from src.scripts.daily_merge import main as daily_merge_main


def main():
    # 1) Finn og hent inneværende sesong
    first = next(iter(LEAGUES))
    cfg = LEAGUES[first]
    base_url = f"https://fbref.com/en/comps/{cfg['comp_id']}/{cfg['slug']}"
    current = get_current_season(base_url)
    if not current:
        raise RuntimeError("Kunne ikke finne gjeldende sesong")
    print(f"=== Fetching raw data for current season: {current} ===")
    fetch_main(seasons=[current])

    # 2) Slå sammen rådata fra forrige sesong med dagens sesong
    print("=== Merging historical data with current season ===")
    daily_merge_main()

    # 3) Definer stat-vinduer og kataloger
    data_dir = "data"
    models_dir = os.path.join(data_dir, "models")
    stat_windows = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}

    # 4) Process og train for hver liga
    for league_name in LEAGUES:
        key = league_name.lower().replace(" ", "_")
        raw_file = os.path.join(data_dir, "raw", f"{key}_matches_full.csv")
        if not os.path.exists(raw_file):
            print(
                f"[WARN] Raw data not found for {league_name} (expected at {raw_file}), skipping."
            )
            continue

        print(f"\n--- Processing data for league: {league_name} ---")
        df_all = pd.read_csv(raw_file, parse_dates=["date"])
        df_processed = process_matches(df_all, stat_windows, league_name)

        # Build feature lists for trening
        features_home = (
            [f"xg_home_roll{w}" for w in stat_windows["xg"]]
            + [f"gf_home_roll{w}" for w in stat_windows["gf"]]
            + [f"xg_conceded_away_roll{w}" for w in stat_windows["xg"]]
            + [f"ga_away_roll{w}" for w in stat_windows["ga"]]
            + ["avg_goals_for_home", "avg_goals_against_away"]
        )

        features_away = (
            [f"xg_away_roll{w}" for w in stat_windows["xg"]]
            + [f"gf_away_roll{w}" for w in stat_windows["gf"]]
            + [f"xg_conceded_home_roll{w}" for w in stat_windows["xg"]]
            + [f"ga_home_roll{w}" for w in stat_windows["ga"]]
            + ["avg_goals_for_away", "avg_goals_against_home"]
        )

        print(f"--- Training models for league: {league_name} ---")
        train_league(
            league_name=league_name,
            data_dir=data_dir,
            models_dir=models_dir,
            features_home=features_home,
            features_away=features_away,
        )

    print("\n=== All leagues processed and models trained ===")


if __name__ == "__main__":
    main()
