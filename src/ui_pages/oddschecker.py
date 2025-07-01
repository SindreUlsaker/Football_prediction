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


def show_feature_weights(model, feature_names):
    """Viser kun relativ betydning av hver feature i prosent."""
    # Map originale features til brukervennlige navn
    name_map = {
        "xg_roll5": "xG siste 5 kamper (lag)",
        "xg_roll10": "xG siste 10 kamper (lag)",
        "gf_roll5": "Mål scoret siste 5 kamper (lag)",
        "gf_roll10": "Mål scoret siste 10 kamper (lag)",
        "avg_goals_for": "Gj.snittsmål scoret (lag)",
        "xg_conceded_away_roll5": "xG motstander siste 5 kamper",
        "xg_conceded_away_roll10": "xG motstander siste 10 kamper",
        "ga_away_roll5": "Inslupne mål motstander siste 5 kamper",
        "ga_away_roll10": "Inslupne mål motstander siste 10 kamper",
        "avg_goals_against_away": "Gj.snitts mål sluppet inn av motstander",
        "is_home": "Spiller på hjemmebane?",
    }

    # Hvilke features som tilhører lag vs motstander
    team_features = [
        "xg_roll5",
        "xg_roll10",
        "gf_roll5",
        "gf_roll10",
        "avg_goals_for",
        "is_home",
    ]
    opponent_features = [
        "xg_conceded_away_roll5",
        "xg_conceded_away_roll10",
        "ga_away_roll5",
        "ga_away_roll10",
        "avg_goals_against_away",
    ]

    # Hent ut koeffisientene fra modellen
    coefs = model.coef_.flatten()

    # Hvis modellen har én ekstra koeff (is_home) som ikke er i feature_names, legg den til
    if len(coefs) == len(feature_names) + 1 and "is_home" not in feature_names:
        feature_names = feature_names + ["is_home"]

    # Fjern dubletter, behold rekkefølge
    feature_names = list(dict.fromkeys(feature_names))

    # Bygg DataFrame med rå og absolutte vekter
    df = pd.DataFrame({"Feature": feature_names, "Weight": coefs})
    df["AbsWeight"] = df["Weight"].abs()
    df["Navn"] = df["Feature"].map(name_map)
    total = df["AbsWeight"].sum()
    df["Betydning (%)"] = (df["AbsWeight"] / total * 100).round(1)

    # Vis i Streamlit uten indeks
    st.subheader("📊 Relativ feature‐betydning")

    st.markdown("**Lagets egne features**")
    df_team = df[df["Feature"].isin(team_features)][
        ["Navn", "Betydning (%)"]
    ].reset_index(drop=True)
    st.dataframe(df_team, hide_index=True, use_container_width=True)

    st.markdown("**Motstander‐features**")
    df_opp = df[df["Feature"].isin(opponent_features)][
        ["Navn", "Betydning (%)"]
    ].reset_index(drop=True)
    st.dataframe(df_opp, hide_index=True, use_container_width=True)


def load_upcoming_matches(league_name: str) -> pd.DataFrame:
    key = league_name.lower().replace(" ", "_")
    processed_path = f"{DATA_PATH}/processed/{key}_processed.csv"
    df = pd.read_csv(processed_path, parse_dates=["date"])

    # For test: hardkodet 16. mai 2025
    now = pd.Timestamp("2025-05-16")
    next_week = now + timedelta(days=10)
    return df[
        (df["date"] >= now) & (df["date"] < next_week) & (df["result_home"].isna())
    ].sort_values("date")


def show_odds_checker():
    st.title("Odds Checker 🔍")

    league = st.selectbox("Velg liga", list(LEAGUES.keys()), key="odds_match")
    matches = load_upcoming_matches(league)
    if matches.empty:
        st.warning("Ingen kommende kamper funnet.")
        return

    matches["label"] = (
        matches["home_team"]
        + " - "
        + matches["away_team"]
        + " ("
        + matches["date"].dt.strftime("%Y-%m-%d %H:%M")
        + ")"
    )
    sel = st.selectbox("Velg kamp", matches["label"], key="odds_match_select")
    selected = matches[matches["label"] == sel].iloc[[0]].reset_index(drop=True)

    spill = st.selectbox(
        "Velg spilltype", ["HUB", "Begge lag scorer", "Over/Under"], key="odds_type"
    )

    # Bygg feature-lister akkurat som før
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

    # Kall relevant odds-funksjon
    if spill == "HUB":
        df_odds = calculate_hub_odds(
            selected,
            features_home,
            features_away,
            league,
            models_dir=f"{DATA_PATH}/models",
        )
        st.subheader("Fair odds for 1X2")
    elif spill == "Begge lag scorer":
        df_odds = calculate_btts_odds(
            selected,
            features_home,
            features_away,
            league,
            models_dir=f"{DATA_PATH}/models",
        )
        st.subheader("Fair odds for Begge lag scorer")
    else:  # Over/Under
        # Du kan la brukeren velge threshold, f.eks.:
        threshold = st.number_input("Sett mål-grense", value=2.5, step=0.5)
        df_odds = calculate_over_under_odds(
            selected,
            features_home,
            features_away,
            league,
            models_dir=f"{DATA_PATH}/models",
            threshold=threshold,
        )
        st.subheader(f"Fair odds for Over/Under {threshold}")

    st.dataframe(df_odds, use_container_width=True, hide_index=True)
