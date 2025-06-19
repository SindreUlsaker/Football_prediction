#!/usr/bin/env python
"""
Script for fetching raw data and processing it for all leagues.
Plasser denne filen i src/data/update_data.py og kjør den fra prosjektets rot.
"""
import pandas as pd
from config.leagues import LEAGUES
from src.data.fetch import main as fetch_main
from src.data.process import process_matches


def main():
    # 1) Fetch raw data for all leagues
    print("=== Henter rådata for alle ligaer ===")
    fetch_main()

    # 2) Process raw data to prosessert CSV for hver liga
    print("=== Behandler rådata og lagrer prosesserte filer ===")
    stat_windows = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}
    for league_name in LEAGUES:
        key = league_name.lower().replace(" ", "_")
        raw_path = f"data/raw/{key}_matches_full.csv"
        try:
            df_all = pd.read_csv(raw_path, parse_dates=["date"])
            process_matches(df_all, stat_windows, league_name)
        except FileNotFoundError:
            print(f"Rådata ikke funnet for {league_name}: {raw_path}")
        except Exception as e:
            print(f"Feil ved behandling av {league_name}: {e}")

    print("=== Ferdig! ===")


if __name__ == "__main__":
    main()
