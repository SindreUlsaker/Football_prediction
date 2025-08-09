#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from src.data.fetch import get_current_season, get_prev_season
from config.leagues import LEAGUES


def main():
    raw_dir = Path("data") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for league_name, cfg in LEAGUES.items():
        key = league_name.lower().replace(" ", "_")
        base_url = f"https://fbref.com/en/comps/{cfg['comp_id']}/{cfg['slug']}"
        current = get_current_season(base_url)
        prev = get_prev_season(current)

        # Fil-stier for sesongene
        prev_file = raw_dir / f"{key}_{prev}_matches.csv"
        current_file = raw_dir / f"{key}_{current}_matches.csv"
        combined_file = raw_dir / f"{key}_matches_full.csv"

        # Les inn rådata for hver sesong (hvis de finnes)
        parts = []
        if prev_file.exists():
            parts.append(pd.read_csv(prev_file, parse_dates=["date"]))
        else:
            print(f"[WARN] Ingen fil for forrige sesong: {prev_file}")
        if current_file.exists():
            parts.append(pd.read_csv(current_file, parse_dates=["date"]))
        else:
            raise FileNotFoundError(f"Mangler dagens rådata: {current_file}")

        # Slå sammen og fjern duplikater
        df_full = pd.concat(parts, ignore_index=True)
        df_full = df_full.drop_duplicates(subset=["date", "team", "opponent"])
        
        df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")
        df_full["date"] = df_full["date"].dt.strftime("%Y-%m-%d")

        # Skriv én samlet fil
        df_full.to_csv(combined_file, index=False)
        print(f"[INFO] Kombinert rådata skrevet til {combined_file}")


if __name__ == "__main__":
    main()
