#!/usr/bin/env python
"""
Script to fetch raw data, process data, and train models for all leagues.
"""
import os
import pandas as pd
from config.leagues import LEAGUES
from src.data.fetch import main as fetch_main
from src.data.process import process_matches
from src.models.train import train_league


def main():
    # 1) Fetch raw data for all leagues
    print("=== Fetching raw data for all leagues ===")
    #fetch_main()

    # 2) Define directories and stat windows
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = "data"
    models_dir = os.path.join(data_dir, "models")
    stat_windows = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}

    # 3) Process and train for each league
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

        # Build feature lists (must match training in pipeline)
        features_home = (
            [f"xg_home_roll{w}" for w in stat_windows["xg"]]
            + [f"gf_home_roll{w}" for w in stat_windows["gf"]]
            + [f"xg_conceded_away_roll{w}" for w in stat_windows["xg"]]
            + [f"ga_away_roll{w}" for w in stat_windows["ga"]]
            + ["avg_goals_for_home", "avg_goals_against_away"]
        )

        # Bortelag
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
