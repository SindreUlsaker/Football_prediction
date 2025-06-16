import streamlit as st
from config.settings import DATA_PATH
from src.ui_components.widgets import round_input, run_button
from src.ui_components.display import show_predictions
from src.pipeline import run_pipeline


def main():
    st.title("Poisson-prediksjoner")
    runde = round_input(min_value=1, max_value=38, value=1)
    if run_button("Kjør prediksjon"):
        preds = run_pipeline(runde, DATA_PATH)
        if preds.empty:
            st.write("Ingen kamper å predikere for denne runden.")
        else:
            show_predictions(preds)
