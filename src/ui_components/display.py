# File: src/ui_components/display.py
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components


def show_predictions(df: pd.DataFrame, prediction_type: int):
    """
    Viser prediksjoner kun med relevante kolonner og forbedrede kolonnenavn.
    """

    df = df.copy()
    # --- DATO / TID ---
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.strftime("%A %Y-%m-%d")
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"]).dt.strftime("%H:%M")
        except Exception:
            pass
    else:
        df["time"] = df["date"].dt.strftime("%H:%M")

    df = df.sort_values(by=["date", "time"]).reset_index(drop=True)
    df.index = df.index + 1  # nummerering

    if prediction_type == 0:
        # Sannsynlighet
        display_df = df[
            ["time", "home_team", "away_team", "prob_home", "prob_draw", "prob_away"]
        ].copy()
        display_df = display_df.rename(
            columns={
                "time": "Time",
                "home_team": "Home team",
                "away_team": "Away team",
                "prob_home": "Home win probability",
                "prob_draw": "Draw probability",
                "prob_away": "Away win probability",
            }
        )
        fmt = {
            "Home win probability": "{:.1%}",
            "Draw probability": "{:.1%}",
            "Away win probability": "{:.1%}",
        }   
        st.subheader("📅 Prediksjoner gruppert per dag – Sannsynlighet")
    else:
        # Fair Odds = 1 / sannsynlighet
        # Unngå deling på null ved å sette et høyt tall hvis prob == 0
        df["odds_home"] = (
            df["prob_home"]
            .replace(0, pd.NA)
            .apply(lambda p: round(1 / p, 2) if pd.notna(p) else None)
        )
        df["odds_draw"] = (
            df["prob_draw"]
            .replace(0, pd.NA)
            .apply(lambda p: round(1 / p, 2) if pd.notna(p) else None)
        )
        df["odds_away"] = (
            df["prob_away"]
            .replace(0, pd.NA)
            .apply(lambda p: round(1 / p, 2) if pd.notna(p) else None)
        )

        display_df = df[
            ["time", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]
        ].copy()
        display_df = display_df.rename(
            columns={
                "time": "Time",
                "home_team": "Home team",
                "away_team": "Away team",
                "odds_home": "Home win fair odds",
                "odds_draw": "Draw fair odds",
                "odds_away": "Away win fair odds",
            }
        )
        fmt = {
            "Home win fair odds": "{}",
            "Draw fair odds": "{}",
            "Away win fair odds": "{}",
        }
        st.subheader("📅 Prediksjoner gruppert per dag – Fair Odds")

    # --- VIS HVER DAG I EXPANDER ---
    for day, group in df.groupby("day", sort=False):
        with st.expander(day, expanded=True):
            # Vis kun relevante kolonner med nye navn
            table = display_df[df["day"] == day]
            st.dataframe(
                table.style.format(fmt), hide_index=True, use_container_width=True
            )
        st.markdown("")  # litt luft under hver expander


def show_odds(df: pd.DataFrame):
    st.dataframe(df, use_container_width=True, hide_index=True)
