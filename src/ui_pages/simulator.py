# File: src/ui_pages/simulator.py
import os
import pandas as pd
import streamlit as st
from config.leagues import LEAGUES
from config.settings import DATA_PATH


def _sim_path(league: str) -> str:
    key = league.lower().replace(" ", "_")
    return f"{DATA_PATH}/processed/simulations/{key}_sim.csv"


def show_simulator_page_cached():
    st.subheader("Liga-simulator (forhåndsberegnet)")

    league = st.selectbox("Liga", list(LEAGUES.keys()), key="sim_league_cached")
    path = _sim_path(league)

    if not os.path.exists(path):
        st.warning(
            "Ingen forhåndsberegnede simuleringer funnet ennå. Kom tilbake etter neste daglige oppdatering."
        )
        return

    df = pd.read_csv(path)

    # Metadata (vises hvis til stede)
    meta_cols = ["Season", "N_sims", "GeneratedAtUTC"]
    if all(c in df.columns for c in meta_cols):
        season = df["Season"].iloc[0]
        n_sims = int(df["N_sims"].iloc[0])
        ts_raw = df["GeneratedAtUTC"].iloc[0]

        ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        try:
            ts_local = ts.tz_convert("Europe/Oslo")
        except Exception:
            ts_local = ts  # fallback
        ts_disp = ts_local.strftime("%d.%m.%Y %H:%M")

        st.caption(
            f"Sesong: **{season}** • Simuleringer: **{n_sims}** • Generert: **{ts_disp}**"
        )

    # Fjern meta-kolonner fra hovedtabellen
    drop_cols = ["League", "Season", "N_sims", "GeneratedAtUTC"]
    cols = [c for c in df.columns if c not in drop_cols]
    display_df = df[cols].copy()
    # Sorter med tie-breakers: P(vinne) -> P(topp 5) -> P(nedrykk)
    display_df = display_df.sort_values(
        by=["P(vinne)", "P(topp 5)", "P(nedrykk)"],
        ascending=[False, False, True],
        kind="mergesort",
    )

    # Konverter sannsynlighetskolonner (float i [0,1]) til prosent
    num_cols = display_df.select_dtypes(include="number").columns.tolist()
    prob_cols = []
    for c in num_cols:
        col = display_df[c]
        try:
            if col.notna().any():
                mn, mx = col.min(), col.max()
                if pd.notna(mn) and pd.notna(mx) and 0.0 <= mn and mx <= 1.0:
                    prob_cols.append(c)
        except Exception:
            pass
    for c in prob_cols:
        display_df[c] = (display_df[c] * 100).round(1).astype(str) + "%"

    st.dataframe(display_df, hide_index=True, use_container_width=True)
