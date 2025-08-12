# File: src/scripts/simulate_all.py
import os
import argparse
import pandas as pd
from datetime import datetime, timezone
from config.leagues import LEAGUES
from config.settings import DATA_PATH
from src.models.simulate import run_simulations


def _processed_path(league: str) -> str:
    key = league.lower().replace(" ", "_")
    return f"{DATA_PATH}/processed/{key}_processed.csv"


def _latest_season_from_file(league: str) -> str:
    df = pd.read_csv(_processed_path(league))

    def keyfunc(s: str) -> int:
        try:
            return int(str(s).split("-")[0])
        except Exception:
            return -1

    return sorted(df["season"].dropna().unique(), key=keyfunc)[-1]


def _save_simulation(
    league: str, season: str, n_sims: int, df_out: pd.DataFrame
) -> str:
    out_dir = f"{DATA_PATH}/processed/simulations"
    os.makedirs(out_dir, exist_ok=True)
    key = league.lower().replace(" ", "_")
    out_path = f"{out_dir}/{key}_sim.csv"
    df_save = df_out.copy()
    df_save.insert(0, "League", league)
    df_save.insert(1, "Season", season)
    df_save.insert(2, "N_sims", n_sims)
    df_save.insert(
        3, "GeneratedAtUTC", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    df_save.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Simulate all leagues and save results"
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=5000,
        help="Antall simuleringer per liga (default 5000)",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Topp-N terskel")
    parser.add_argument(
        "--relegation-spots", type=int, default=3, help="Antall nedrykksplasser"
    )
    args = parser.parse_args()

    for league in LEAGUES.keys():
        try:
            season = _latest_season_from_file(league)
            res = run_simulations(
                league_name=league,
                n_sims=args.n_sims,
                season=season,
                top_n=args.top_n,
                relegation_spots=args.relegation_spots,
                models_dir=f"{DATA_PATH}/models",
            )
            out_path = _save_simulation(league, season, args.n_sims, res)
            print(f"[SIM] {league} ({season}) → {out_path}")
        except Exception as e:
            print(f"[SIM][WARN] Skipped {league}: {e}")


if __name__ == "__main__":
    main()
