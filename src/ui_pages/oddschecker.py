# File: src/ui_pages/oddschecker.py
import streamlit as st
import pandas as pd
from datetime import timedelta
from config.leagues import LEAGUES
from config.settings import DATA_PATH
from src.models.odds import (
    calculate_hub_odds,
    calculate_btts_odds,
    calculate_over_under_odds,
)

import pandas as pd
from src.models.predict import load_models_for_league

def load_upcoming_matches(league_name: str) -> pd.DataFrame:
    key = league_name.lower().replace(" ", "_")
    processed_path = f"{DATA_PATH}/processed/{key}_processed.csv"
    df = pd.read_csv(processed_path, parse_dates=["date"])

    now = pd.Timestamp.now()
    next_week = now + timedelta(days=18)
    return df[
        (df["date"] >= now) & (df["date"] < next_week) & (df["result_home"].isna())
    ].sort_values("date")


def show_odds_checker(
    matches: pd.DataFrame,
    odds_type: str,
    threshold: float | None,
    show_weights: bool,
    league: str,
    sel_label: str,
):
    """Odds Checker-side: sidebar for alle valg og hovedvisning under."""

    # --- HURTIGMETRIKKER ---
    m1, m2 = st.columns([3, 1])
    m1.metric("🏷️ Valgt kamp", sel_label)
    m2.metric("🎲 Spilltype", odds_type)
    st.markdown("---")

    # Finn den valgte kampen
    sel_match = matches[matches["label"] == sel_label].iloc[[0]].reset_index(drop=True)

    # Bygg feature-lister (uendret logikk)
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

    # Beregn og vis de ulike odds-tabellene, uten å endre logikken
    if odds_type == "HUB":
        df_odds = calculate_hub_odds(
            sel_match,
            features_home,
            features_away,
            league,
            models_dir=f"{DATA_PATH}/models",
        )
        st.markdown("### Fair odds 1X2")
    elif odds_type == "Begge lag scorer":
        df_odds = calculate_btts_odds(
            sel_match,
            features_home,
            features_away,
            league,
            models_dir=f"{DATA_PATH}/models",
        )
        st.markdown("### Fair odds - Begge lag scorer")
    else:
        df_odds = calculate_over_under_odds(
            sel_match,
            features_home,
            features_away,
            league,
            models_dir=f"{DATA_PATH}/models",
            threshold=threshold,
        )
        st.markdown(f"### Fair odds - Over/Under {threshold}")

    # Vis resultat
    st.dataframe(df_odds, use_container_width=True, hide_index=True)
