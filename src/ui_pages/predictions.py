# File: src/ui_pages/predictions.py
import streamlit as st
import pandas as pd
from datetime import timedelta
from config.leagues import LEAGUES
from config.settings import DATA_PATH
from src.models.predict import load_models_for_league, predict_poisson_from_models
from src.ui_components.display import show_predictions

# Stat window configuration (samme som ved trening)
STAT_WINDOWS = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}


def load_upcoming_matches(league_name: str) -> pd.DataFrame:
    """
    Leser ferdig prosessert data og returnerer kamper som spilles i den kommende uken.
    """
    key = league_name.lower().replace(" ", "_")
    processed_path = f"{DATA_PATH}/processed/{key}_processed.csv"
    df = pd.read_csv(processed_path, parse_dates=["date"])

    now = pd.Timestamp.now()
    next_week = now + timedelta(days=18)
    df_upcoming = df[
        (df["date"] >= now) & (df["date"] < next_week) & (df["result_home"].isna())
    ].copy()
    return df_upcoming.sort_values("date")


def show_predictions_page(
    league: str, vis_type: str = "Sannsynlighet"
):
    # --- LAST INN OG SJEKK DATA ---
    matches = load_upcoming_matches(league)
    if matches.empty:
        st.warning("Ingen kommende kamper funnet for den neste uken.")
        return

    # --- BYGG FEATURES (uendret logikk) ---
    features_home = (
        [f"xg_home_roll{w}" for w in STAT_WINDOWS["xg"]]
        + [f"gf_home_roll{w}" for w in STAT_WINDOWS["gf"]]
        + [f"xg_conceded_away_roll{w}" for w in STAT_WINDOWS["xg"]]
        + [f"ga_away_roll{w}" for w in STAT_WINDOWS["ga"]]
        + ["avg_goals_for_home", "avg_goals_against_away"]
    )
    features_away = (
        [f"xg_away_roll{w}" for w in STAT_WINDOWS["xg"]]
        + [f"gf_away_roll{w}" for w in STAT_WINDOWS["gf"]]
        + [f"xg_conceded_home_roll{w}" for w in STAT_WINDOWS["xg"]]
        + [f"ga_home_roll{w}" for w in STAT_WINDOWS["ga"]]
        + ["avg_goals_for_away", "avg_goals_against_home"]
    )

    # --- PREDIKSJON (uendret) ---
    preds = predict_poisson_from_models(
        df=matches,
        features_home=features_home,
        features_away=features_away,
        league_name=league,
        models_dir=f"{DATA_PATH}/models",
        max_goals=10,
    )

    # --- HURTIGMETRIKKER ---
    m1, m2 = st.columns(2)
    m1.metric("🗓 Antall kamper", len(preds))
    avg_goals = (preds["lambda_home"] + preds["lambda_away"]).mean().round(2)
    m2.metric("⚽ Snittmål", avg_goals)

    st.markdown("---")

    # --- VISNING ---
    mode = 0 if vis_type == "Sannsynlighet" else 1
    show_predictions(preds, mode)
