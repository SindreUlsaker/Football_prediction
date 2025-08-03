# File: src/ui_pages/main.py
import pandas as pd
import streamlit as st
from config.leagues import LEAGUES
from src.ui_pages.predictions import show_predictions_page
from src.ui_pages.oddschecker import (
    show_odds_checker,
    load_upcoming_matches as load_odds,
)
from src.ui_pages.model_info import show_model_info_page
import base64


def get_base64_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def main():

    # --- SENTRERT BANNER (som før) ---
    logo_base64 = get_base64_image("assets/logo.png")
    st.markdown(
        f'''
        <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
          <img src="data:image/png;base64,{logo_base64}" width="60" />
          <div style="text-align: center;">
            <h3 style="margin:0;">Fotballprediksjoner</h3>
            <p style="color:#555; margin:0;">AI-basert analyse</p>
          </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # --- VIS FANENE (disse må stå utenfor kolonner) ---
    tab_preds, tab_odds, tab_info = st.tabs(
        ["Prediksjoner (kommende uke)", "Odds Checker", "Modellinfo"]
    )

    # --- LINJE UNDER FANENE OVER HELE BREDDE ---
    st.markdown("<hr style='margin-top: -1px;'>", unsafe_allow_html=True)


    # --- PREDIKSJONER TAB ---
    with tab_preds:
        left, right = st.columns([1, 4], gap="large")
        with left:
            st.header("Filter")
            league = st.selectbox("Velg liga", list(LEAGUES.keys()), key="preds_league")
            vis_type = st.radio("Visingsmodus", ["Sannsynlighet", "Fair Odds"], index=0)
            use_date_filter = st.checkbox("Filtrer på én dato", key="preds_date_filter")
            selected_date = None
            if use_date_filter:
                selected_date = st.date_input("Velg dato", value=pd.to_datetime("today").date(), key="preds_date")
        with right:
            show_predictions_page(league, vis_type, selected_date)

    # --- ODDS CHECKER TAB ---
    with tab_odds:
        left, right = st.columns([1, 4], gap="large")
        with left:
            st.header("Filter")
            league = st.selectbox("Velg liga", list(LEAGUES.keys()), key="odds_league")
            use_date_filter = st.checkbox("Filtrer på én dato", key="odds_date_filter")
            selected_date = None
            if use_date_filter:
                selected_date = st.date_input("Velg dato", value=pd.to_datetime("today").date(), key="odds_date")
            matches = load_odds(league, selected_date)
            if matches.empty:
                st.warning("Ingen kommende kamper funnet for valgt liga.")
                sel_match = odds_type = threshold = show_weights = None
            else:
                matches["time_clean"] = matches["time"].str.extract(r"(\d{1,2}:\d{2})")[
                    0
                ]
                matches["dt"] = pd.to_datetime(
                    matches["date"].dt.strftime("%Y-%m-%d")
                    + " "
                    + matches["time_clean"]
                )
                matches["label"] = (
                    matches["home_team"]
                    + " vs "
                    + matches["away_team"]
                    + " ("
                    + matches["dt"].dt.strftime("%d.%m %H:%M")
                    + ")"
                )
                sel_label = st.selectbox(
                    "Velg kamp", matches["label"].tolist(), key="odds_match"
                )
                odds_type = st.selectbox(
                    "Visningsmodus spilltype",
                    ["HUB", "Begge lag scorer", "Over/Under"],
                    key="odds_type",
                )
                threshold = None
                if odds_type == "Over/Under":
                    threshold = st.number_input(
                        "Sett Over/Under-grense", value=2.5, step=1.0
                    )
                show_weights = st.checkbox(
                    "Vis modellens feature-vekter", key="odds_weights"
                )
                sel_match = (
                    matches[matches["label"] == sel_label]
                    .iloc[[0]]
                    .reset_index(drop=True)
                )
        with right:
            if sel_match is not None:
                show_odds_checker(
                    matches=sel_match,
                    odds_type=odds_type,
                    threshold=threshold,
                    show_weights=show_weights,
                    league=league,
                    sel_label=sel_label,
                )
            else:
                st.info("Velg liga og kamp i venstre panel for å se odds.")

    with tab_info:
        show_model_info_page()

if __name__ == "__main__":
    main()
