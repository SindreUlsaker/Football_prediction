# File: src/ui_components/display.py
import streamlit as st
import pandas as pd


def show_predictions(df: pd.DataFrame):
    df = df.copy()
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    # Extract day for grouping
    df["day"] = df["date"].dt.strftime("%A %Y-%m-%d")
    # Use existing 'time' column if present
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"]).dt.strftime("%H:%M")
        except Exception:
            pass
    else:
        # Fallback: derive time from date if no time column
        df["time"] = df["date"].dt.strftime("H:%M")

    # Sort by date then time
    df = df.sort_values(by=["date", "time"])

    # Number matches starting at 1
    df = df.reset_index(drop=True)
    df.index = df.index + 1

    # Drop original 'date' and 'day' columns
    display_df = df.drop(columns=["date", "day"])

    # Format float columns
    float_cols = ["lambda_home", "lambda_away", "prob_home", "prob_draw", "prob_away"]
    format_dict = {col: "{:.3f}" for col in float_cols if col in display_df.columns}

    # Group by day and display styled table
    for day, group in df.groupby("day"):
        st.markdown(f"**{day}**")
        cols = ["time"] + [col for col in display_df.columns if col != "time"]
        table = group[cols]
        st.dataframe(table.style.format(format_dict))
