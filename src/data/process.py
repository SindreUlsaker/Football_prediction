import pandas as pd
from src.features.features import add_all_features

def preprocess_data(df_all):
    """
    Preprocess dataset to one row per match (home perspective):
      - Filter to Premier League
      - Drop duplicates and missing
      - Parse date and sort
      - Keep only home matches
      - Rename columns to home_/away_
      - Convert round to numeric
      - Compute result_home (1: home win, 0: draw, -1: away win)
    """
    # 1) Filter Premier League
    df = df_all[df_all["comp"] == "Premier League"].copy()
    # 2) Drop duplicates and missing values
    df = df.drop_duplicates(subset=["date", "team", "opponent"])
    df = df.dropna(subset=["date", "team", "opponent"])

    # 3) Parse date and sort chronologically
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.sort_values("date")

    # 4) Keep only home matches (venue == 'H' or 'Home')
    df_home = df[df["venue"].isin(["H", "Home"])].copy()

    # 5) Rename metadata and stats columns
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

    # --- Normaliser lagnavn før videre prosessering ---
    # 1) Stripp bort whitespace
    df_home["home_team"] = df_home["home_team"].str.strip()
    df_home["away_team"] = df_home["away_team"].str.strip()

    # 2) Erstatt vanlige alias med offisielle navn
    name_map = {
        "Manchester Utd": "Manchester United",
        "Wolves": "Wolverhampton Wanderers",
        "Tottenham": "Tottenham Hotspur",
        "West Ham": "West Ham United",
        "Nott'ham Forest": "Nottingham Forest",
        "Newcastle Utd": "Newcastle United",
        "Brighton": "Brighton and Hove Albion",
    }
    df_home["home_team"] = df_home["home_team"].replace(name_map)
    df_home["away_team"] = df_home["away_team"].replace(name_map)

    # 6) Convert round text (e.g., 'Matchweek 38') to numeric
    df_home["round"] = df_home["round"].astype(str).str.extract(r"(\d+)").astype(int)

    # 7) Compute match outcome from home perspective
    df_home["result_home"] = df_home.apply(
    lambda r: (
        1 if pd.notna(r.gf_home) and pd.notna(r.gf_away) and r.gf_home > r.gf_away else
        0 if pd.notna(r.gf_home) and pd.notna(r.gf_away) and r.gf_home == r.gf_away else
        -1 if pd.notna(r.gf_home) and pd.notna(r.gf_away) and r.gf_home < r.gf_away else
        pd.NA
    ),
    axis=1,
)

    # 8) Select final columns
    cols = [
        "date",
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
    Konverterer oppgitte kolonner til numerisk, tvinger str→NaN om nødvendig.
    """
    df[cols] = df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    return df


def process_matches(
    df: pd.DataFrame, stat_windows: dict[str, list[int]]
) -> pd.DataFrame:
    """
    Full data pipeline:
      1) Preprocess raw DataFrame (one row per match)
      2) Ensure numeric types for goals & result
      3) Add all features via add_all_features
    """
    # 1) Cleaning & basic transforms
    df = preprocess_data(df)

    # 2) Dtype safety for aggregation
    numeric_cols = ["gf_home", "ga_home", "gf_away", "ga_away", "result_home"]
    df = ensure_numeric(df, numeric_cols)

    # 3) Feature-engineering (venue-agnostic)
    df = add_all_features(df, stat_windows)

    return df
