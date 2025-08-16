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
    import numpy as np

    def _coerce_prob(series: pd.Series) -> pd.Series:
        """
        Konverterer '12.3', '12,3', '12.3%', '0.123' → float i [0,1].
        Rører ikke andre kolonner.
        """
        s = series.astype(str).str.strip()
        has_pct = s.str.contains("%").any()
        s = s.str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
        vals = pd.to_numeric(s, errors="coerce")
        if has_pct or (
            vals.max(skipna=True) is not np.nan and vals.max(skipna=True) > 1.001
        ):
            vals = vals / 100.0
        return vals

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
        ts = df["GeneratedAtUTC"].iloc[0]
        st.caption(
            f"Sesong: **{season}** • Simuleringer: **{n_sims}** • Generert: **{ts}**"
        )

    # Fjern meta-kolonner fra hovedtabellen
    drop_cols = ["League", "Season", "N_sims", "GeneratedAtUTC"]
    cols = [c for c in df.columns if c not in drop_cols]

    # Tving sannsynlighetskolonnene til floats slik at sekundær/tertiærsortering fungerer
    prob_cols = ["P(vinne)", "P(topp 5)", "P(nedrykk)"]
    for c in prob_cols:
        if c in df.columns:
            df[c] = _coerce_prob(df[c])

    # Sorter: primært P(vinne) (desc), deretter P(topp 5) (desc), så P(nedrykk) (asc)
    if all(c in df.columns for c in prob_cols):
        df = df.sort_values(
            by=["P(vinne)", "P(topp 5)", "P(nedrykk)"],
            ascending=[False, False, True],
            kind="mergesort",  # stabil; ved fullt likhet beholdes input-rekkefølgen
        )

    st.dataframe(df[cols], hide_index=True, use_container_width=True)
