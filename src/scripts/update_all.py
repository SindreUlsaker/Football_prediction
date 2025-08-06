#!/usr/bin/env python
"""
Script to fetch raw data, process data, and train models for all leagues.
"""
import os
import pandas as pd
from config.leagues import LEAGUES
from src.data.fetch import main as fetch_main, get_current_season
from src.data.process import process_matches
from src.models.train import train_league
from src.scripts.daily_merge import main as daily_merge_main
import argparse


def main():
    # 1) Fetch raw data for all leagues
    parser = argparse.ArgumentParser(description="Fetch, process & train all leagues")
    parser.add_argument(
        "-s",
        "--seasons",
        nargs="+",
        help="Sesonger som skal hentes, f.eks. 2024-2025 eller 2023-2024 2024-2025",
    )
    parser.add_argument(
        "-c", "--only-current", action="store_true", help="Hent bare gjeldende sesong"
    )
    args = parser.parse_args()

    seasons_to_fetch = None
    if args.seasons:
        seasons_to_fetch = args.seasons
    elif args.only_current:
        # Finn gjeldende sesong basert på første liga i konfig
        first = next(iter(LEAGUES))
        cfg = LEAGUES[first]
        base_url = f"https://fbref.com/en/comps/{cfg['comp_id']}/{cfg['slug']}"
        current = get_current_season(base_url)
        if not current:
            raise RuntimeError("Kunne ikke finne gjeldende sesong")
        seasons_to_fetch = [current]

    print("=== Fetching raw data for all leagues ===")
    fetch_main(seasons=seasons_to_fetch)
    
    if args.only_current:
        print("=== Merging historical data with current season ===")
        daily_merge_main()

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
