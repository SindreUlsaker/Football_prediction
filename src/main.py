import streamlit as st
from config.settings import DATA_PATH
from config.leagues import LEAGUES
from src.ui_components.widgets import round_input, run_button
from src.ui_components.display import show_predictions
from src.pipeline import run_pipeline


def main():
    st.title("Prediksjoner per liga")

    # 1) Velg liga
    league = st.selectbox("Velg liga", list(LEAGUES.keys()))

    # 2) Velg rundenummer
    runde = round_input(min_value=1, max_value=38, value=1)

    # 3) Kjør prediksjon
    if run_button("Kjør prediksjon"):
        preds = run_pipeline(runde, DATA_PATH, league)
        if preds.empty:
            st.write(f"Ingen kamper å predikere for {league}, runde {runde}.")
        else:
            show_predictions(preds)


if __name__ == "__main__":
    main()
