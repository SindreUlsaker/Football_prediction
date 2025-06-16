# File: src/ui_components/display.py
import streamlit as st
import pandas as pd


def show_predictions(df: pd.DataFrame):
    df = df.copy()
    # Parse and extract day and time
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.strftime("%A %Y-%m-%d")
    df["time"] = df["date"].dt.strftime("%H:%M")

    # Number matches starting at 1
    df = df.reset_index(drop=True)
    df.index = df.index + 1

    # Drop original date column
    display_df = df.drop(columns=["date", "day"])

    # Group by day and display
    for day, group in df.groupby("day"):
        st.markdown(f"**{day}**")
        st.table(group[["time"] + [col for col in display_df.columns if col != "time"]])
