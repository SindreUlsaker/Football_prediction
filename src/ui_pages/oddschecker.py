# File: src/ui_pages/oddschecker.py
import streamlit as st
import pandas as pd
from datetime import timedelta
from config.leagues import LEAGUES
from config.settings import DATA_PATH
from src.models.predict import load_models_for_league, predict_poisson_from_models

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

    # 1) Velg liga
    league = st.selectbox("Velg liga", list(LEAGUES.keys()), key="odds_match")

    # 2) Last inn kommende kamper
    matches = load_upcoming_matches(league)
    if matches.empty:
        st.warning("Ingen kommende kamper funnet.")
        return

    # 3) Velg kamp
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

    # 4) Velg spilltype (her kun HUB = 1X2)
    spill = st.selectbox("Velg spilltype", ["HUB"], key="odds_type")
    if spill != "HUB":
        return

    # 5) Samme stat-vinduer som i train_all_models.py
    stat_windows = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}

    features_home = (
        [f"xg_home_roll{w}" for w in stat_windows["xg"]]
        + [f"gf_home_roll{w}" for w in stat_windows["gf"]]
        + [f"xg_conceded_away_roll{w}" for w in stat_windows["xg"]]
        + [f"ga_away_roll{w}" for w in stat_windows["ga"]]
        + ["avg_goals_for_home", "avg_goals_against_away"]
    )

    # Bortelag
    features_away = (
        [f"xg_away_roll{w}" for w in stat_windows["xg"]]
        + [f"gf_away_roll{w}" for w in stat_windows["gf"]]
        + [f"xg_conceded_home_roll{w}" for w in stat_windows["xg"]]
        + [f"ga_home_roll{w}" for w in stat_windows["ga"]]
        + ["avg_goals_for_away", "avg_goals_against_home"]
    )

    # 6) Last modeller og scalers
    model, scaler = load_models_for_league(league, models_dir=f"{DATA_PATH}/models")

    # 7) Predict

    preds = predict_poisson_from_models(
        df=selected,
        features_home=features_home,
        features_away=features_away,
        league_name=league,
        models_dir=f"{DATA_PATH}/models",
        max_goals=10
    )

    generic_features = [f.replace("_home", "") for f in features_home] + ["is_home"]

    if st.checkbox("Vis hvilke features modellen vektlegger mest"):
        show_feature_weights(model, generic_features)

    # 8) Vis fair odds
    row = preds.iloc[0]
    odds = {
        "Hjemmeseier": (row["prob_home"], 1 / row["prob_home"]),
        "Uavgjort": (row["prob_draw"], 1 / row["prob_draw"]),
        "Borteseier": (row["prob_away"], 1 / row["prob_away"]),
    }
    df_odds = pd.DataFrame(
        [
            {
                "Utfall": k,
                "Sannsynlighet": f"{v[0]*100:.1f}%",
                "Fair odds": f"{v[1]:.2f}",
            }
            for k, v in odds.items()
        ]
    )
    st.subheader("Fair odds for 1X2")
    st.dataframe(df_odds, use_container_width=True, hide_index=True)
