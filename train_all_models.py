#!/usr/bin/env python
import os
import shutil

from config.leagues import LEAGUES
from config.settings import DATA_PATH
from src.models.train import train_league

# Samme stat-windows som ved trening og UI
STAT_WINDOWS = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}

# Bygg feature-lister
features_home = (
    [f"xg_home_roll{w}" for w in STAT_WINDOWS["xg"]]
    + [f"gf_home_roll{w}" for w in STAT_WINDOWS["gf"]]
    + [f"xg_conceded_away_roll{w}" for w in STAT_WINDOWS["xg"]]
    + [f"ga_away_roll{w}" for w in STAT_WINDOWS["ga"]]
    + ["avg_goals_for_home", "avg_goals_against_away"]
)

# Bortelag
features_away = (
    [f"xg_away_roll{w}" for w in STAT_WINDOWS["xg"]]
    + [f"gf_away_roll{w}" for w in STAT_WINDOWS["gf"]]
    + [f"xg_conceded_home_roll{w}" for w in STAT_WINDOWS["xg"]]
    + [f"ga_home_roll{w}" for w in STAT_WINDOWS["ga"]]
    + ["avg_goals_for_away", "avg_goals_against_home"]
)

def main():
    # data_dir er rot for data (*.csv ligger i data/processed/)
    data_dir = DATA_PATH           # vanligvis "data"
    models_dir = os.path.join(data_dir, "models")

    # Fjern gammel models-mappe om den finnes
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)
    os.makedirs(models_dir, exist_ok=True)

    # Tren én modell per liga
    for league in LEAGUES.keys():
        print(f"[INFO] Training model for {league} …")
        train_league(
            league_name=league,
            data_dir=data_dir,
            models_dir=models_dir,
            features_home=features_home,
            features_away=features_away,
        )

    print("[INFO] All leagues trained. Models saved in:", models_dir)

if __name__ == "__main__":
    main()
