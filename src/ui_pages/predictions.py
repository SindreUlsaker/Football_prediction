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

    now = pd.Timestamp("2025-05-16")
    next_week = now + timedelta(days=10)
    df_upcoming = df[
        (df["date"] >= now) & (df["date"] < next_week) & (df["result_home"].isna())
    ].copy()
    return df_upcoming.sort_values("date")


def show_predictions_page():
    st.title("Prediksjoner og Fair Odds for kommende kamper")

    # 1) Velg liga
    league = st.selectbox("Velg liga", list(LEAGUES.keys()), key="preds_league")

    # 2) Les kommende kamper
    matches = load_upcoming_matches(league)
    if matches.empty:
        st.warning("Ingen kommende kamper funnet for den neste uken.")
        return

    # 4) Velg visningstype
    vis_type = st.radio("Visingsmodus", ["Sannsynlighet", "Fair Odds"])

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

    preds = predict_poisson_from_models(
        df=matches,
        features_home=features_home,
        features_away=features_away,
        league_name=league,
        models_dir=f"{DATA_PATH}/models",
        max_goals=10,
    )

    # 6) Vis resultater
    if vis_type == "Sannsynlighet":
        # Bruk felles display-funksjon
        show_predictions(preds, 0)
    else:
        show_predictions(preds, 1)
