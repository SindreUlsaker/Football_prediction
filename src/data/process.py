import os
import pandas as pd
from src.features.features import add_all_features
from config.leagues import LEAGUES


def preprocess_data(df_all: pd.DataFrame, league_name: str) -> pd.DataFrame:
    """
    Preprocess dataset to one row per match (home perspective) for a given league:
      - Map team names according to league-specific mapping
      - Filter to specified league
      - Drop duplicates and missing values
      - Parse date and sort
      - Keep only home matches (venue == 'Home')
      - Rename columns to home_/away_
      - Convert round to numeric
      - Compute result_home (1: home win, 0: draw, -1: away win)
    """
    # Copy raw data
    df = df_all.copy()

    # 1) Map team/opponent names using league-specific mapping
    cfg = LEAGUES.get(league_name, {})
    team_map = cfg.get("team_name_map") or {}
    df["team"] = df["team"].astype(str).str.strip().apply(lambda x: team_map.get(x, x))
    df["opponent"] = (
        df["opponent"].astype(str).str.strip().apply(lambda x: team_map.get(x, x))
    )

    # 2) Filter to the given league
    df = df[df["comp"].astype(str).str.contains(league_name, case=False, na=False)].copy()

    # 3) Drop duplicates and missing values
    df = df.drop_duplicates(subset=["date", "team", "opponent"])
    df = df.dropna(subset=["date", "team", "opponent"])

    # 4) Parse date and sort
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.sort_values("date")

    # 5) Keep only home matches
    df_home = df[df["venue"] == "Home"].copy()

    # 6) Rename metadata and stats columns for home perspective
    df_home = df_home.rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "gf_for": "gf_home",
            "ga_for": "ga_home",
            "xg_for": "xg_home",
            "gf_against": "gf_away",
            "ga_against": "ga_away",
            "xg_against": "xg_away",
        }
    )

    # 7) Convert round text (e.g., 'Matchweek 38') to numeric, coercing errors
    round_series = df_home["round"].astype(str).str.extract(r"(\d+)")[0]
    df_home["round"] = pd.to_numeric(round_series, errors="coerce").astype("Int64")

    # 8) Compute match outcome from home perspective
    def compute_result(row):
        if pd.notna(row.gf_home) and pd.notna(row.gf_away):
            if row.gf_home > row.gf_away:
                return 1
            if row.gf_home == row.gf_away:
                return 0
            return -1
        return pd.NA

    df_home["result_home"] = df_home.apply(compute_result, axis=1)

    # 9) Select final columns
    cols = [
        "date",
        "time",
        "season",
        "comp",
        "round",
        "home_team",
        "away_team",
        "gf_home",
        "ga_home",
        "xg_home",
        "gf_away",
        "ga_away",
        "xg_away",
        "result_home",
    ]
    df_final = df_home[cols]
    return df_final


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric, coercing errors to NaN.
    """
    df[cols] = df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    return df


def process_matches(
    df_all: pd.DataFrame, stat_windows: dict[str, list[int]], league_name: str
) -> pd.DataFrame:
    """
    Full data pipeline for a given league:
      1) Preprocess raw DataFrame (one row per match, home perspective)
      2) Ensure numeric types for aggregated columns
      3) Add all features via add_all_features
    """
    # 1) Cleaning & basic transforms
    df = preprocess_data(df_all, league_name)

    # 2) Dtype safety for aggregation
    numeric_cols = ["gf_home", "ga_home", "gf_away", "ga_away", "result_home"]
    df = ensure_numeric(df, numeric_cols)

    # 3) Feature-engineering
    AGG_WINDOW = 10 # Number of matches to aggregate for static features
    df = add_all_features(df, stat_windows, agg_window=AGG_WINDOW)

    # 4) Lagre ferdig prosessert DataFrame til CSV
    filename = os.path.join("data", "processed", league_name.lower().replace(" ", "_") + "_processed.csv")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Lagret prosessert data til {filename}")
    
    return df
