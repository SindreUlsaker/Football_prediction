#!/usr/bin/env python
"""
compute_kappa.py

Script for å estimere bias‐faktorer (kappa) for Premier League Poisson‐modell.

Kjør fra prosjektroten:
    python src/scripts/compute_kappa.py

Eller pakk inn stier:
    python src/scripts/compute_kappa.py --csv path/to/premier_league_processed.csv \
                                        --models-dir path/to/models
"""

import sys
import os

# —— Legg prosjektroten i sys.path for å kunne importere src-pakken ——
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
# ———————————————————————————————————————————————————————

import argparse
import numpy as np
import pandas as pd

from src.models.predict import (
    load_models_for_league,
)  # :contentReference[oaicite:0]{index=0}


def estimate_bias_factor(y_true: np.ndarray, lambda_pred: np.ndarray) -> float:
    """Returnerer κ = sum(obs mål) / sum(pred λ)."""
    return np.sum(y_true) / np.sum(lambda_pred)


def main():
    parser = argparse.ArgumentParser(
        description="Compute kappa for Premier League model"
    )
    parser.add_argument(
        "--csv",
        default=os.path.join("data", "processed", "premier_league_processed.csv"),
        help="Path til prosessert CSV-fil",
    )
    parser.add_argument(
        "--models-dir",
        default=os.path.join("data", "models"),
        help="Mappe der modellen og scaler er lagret",
    )
    parser.add_argument(
        "--league",
        default="Premier League",
        help="Ligaen som brukes for å laste riktig modell",
    )
    args = parser.parse_args()

    # 1) Les inn ferdig prosessert data
    df = pd.read_csv(args.csv, parse_dates=["date"])
    df = df[df["gf_home"].notna() & df["gf_away"].notna()].copy()

    # 2) Bygg feature-lister i samme stil som update_all.py :contentReference[oaicite:1]{index=1}
    stat_windows = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}
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

    # 3) Last inn modell + scaler
    model, scaler = load_models_for_league(args.league, args.models_dir)

    # 4) Forbered X slik trenings‐pipeline gjorde:
    #    - Fjern både "_home" og "_away" fra *alle* kolonnenavn
    #    - Legg til is_home‐indikator
    Xh = df[features_home].copy()
    Xh.columns = [c.replace("_home", "").replace("_away", "") for c in Xh.columns]
    Xh["is_home"] = 1

    Xa = df[features_away].copy()
    Xa.columns = [c.replace("_home", "").replace("_away", "") for c in Xa.columns]
    Xa["is_home"] = 0

    # 5) Slå sammen, fyll NaN, og sørg for at vi har alle kolonnene scaler forventer
    X_all = pd.concat([Xh, Xa], ignore_index=True).fillna(0)

    # Hvis det er noen features i scaler.feature_names_in_ som mangler i X_all,
    # legg dem inn med verdi 0
    for feat in scaler.feature_names_in_:
        if feat not in X_all.columns:
            X_all[feat] = 0

    # Re-rekkefølge etter akkurat det scaler.feature_names_in_ krever
    X_all = X_all[list(scaler.feature_names_in_)]

    # 6) Skalér og prediker λ
    X_scaled = scaler.transform(X_all)
    lambdas = model.predict(X_scaled)
    lambda_home = lambdas[: len(df)]
    lambda_away = lambdas[len(df) :]

    # 7) Estimer κ‐faktorer og print
    kappa_home = estimate_bias_factor(df["gf_home"].to_numpy(), lambda_home)
    kappa_away = estimate_bias_factor(df["gf_away"].to_numpy(), lambda_away)
    print(f"[RESULT] κ_home = {kappa_home:.3f}, κ_away = {kappa_away:.3f}")


if __name__ == "__main__":
    main()
