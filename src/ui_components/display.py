import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

def show_predictions(df: pd.DataFrame, prediction_type: int):
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
        df["time"] = df["date"].dt.strftime("%H:%M")

    # Sort by date then time
    df = df.sort_values(by=["date", "time"])

    # Number matches starting at 1
    df = df.reset_index(drop=True)
    df.index = df.index + 1

    # Compute fair odds if requested
    if prediction_type == 1:
        # Calculate fair odds columns
        df["fair_odds_home"] = (1 / df.get("prob_home")).round(2)
        df["fair_odds_draw"] = (1 / df.get("prob_draw")).round(2)
        df["fair_odds_away"] = (1 / df.get("prob_away")).round(2)
        # Drop raw probabilities
        display_df = df.drop(
            columns=["date", "day", "prob_home", "prob_draw", "prob_away"]
        )
        # Columns to format as floats
        float_cols = [
            col
            for col in [
                "lambda_home",
                "lambda_away",
                "fair_odds_home",
                "fair_odds_draw",
                "fair_odds_away",
            ]
            if col in display_df.columns
        ]
    else:
        # Keep probabilities as-is
        display_df = df.drop(columns=["date", "day"])
        # Columns to format as floats
        float_cols = [
            col
            for col in [
                "lambda_home",
                "lambda_away",
                "prob_home",
                "prob_draw",
                "prob_away",
            ]
            if col in display_df.columns
        ]

    # Format float columns
    if prediction_type == 1:
        # For fair odds, format to 2 decimal places
        format_dict = {col: "{:.2f}" for col in float_cols}
    else:
        format_dict = {col: "{:.3f}" for col in float_cols}

    # Group by day and display styled table
    for day, group in df.groupby("day"):
        st.markdown(f"**{day}**")
        # Determine order: always show time first
        cols = ["time"] + [c for c in display_df.columns if c != "time"]
        table = group[cols]
        # Display with increased width
        st.dataframe(table.style.format(format_dict), hide_index=True, use_container_width=True)
