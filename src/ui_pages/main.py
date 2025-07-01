# File: src/main.py
import streamlit as st
from src.ui_pages.predictions import show_predictions_page
from src.ui_pages.oddschecker import show_odds_checker


def main():
    st.title("Football Prediction App")

    # Top navigation tabs
    tab_preds, tab_odds = st.tabs(["Prediksjoner (kommende uke)", "Odds Checker"])

    # Tab: Prediksjoner
    with tab_preds:
        show_predictions_page()

    # Tab: Odds Checker
    with tab_odds:
        show_odds_checker()


if __name__ == "__main__":
    main()
