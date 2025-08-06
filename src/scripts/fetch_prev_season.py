#!/usr/bin/env python3
"""
Script to fetch only the previous season's raw data for all configured leagues.
"""
import sys
from src.data.fetch import main as fetch_main, get_current_season, get_prev_season
from config.leagues import LEAGUES


def main():
    # Finn første liga i konfigurasjonen
    first = next(iter(LEAGUES))
    cfg = LEAGUES[first]
    base_url = f"https://fbref.com/en/comps/{cfg['comp_id']}/{cfg['slug']}"

    # Hent gjeldende og forrige sesong
    current = get_current_season(base_url)
    previous = get_prev_season(current)

    if not previous:
        print("Fant ikke forrige sesong, avbryter.")
        sys.exit(1)

    print(f"[INFO] Henter forrige sesong: {previous}")

    # Kjør kun fetch-delen for forrige sesong
    fetch_main(seasons=[previous])


if __name__ == "__main__":
    main()
