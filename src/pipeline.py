import os
import pandas as pd
import joblib
from src.data.process import process_matches
from src.models.predict import load_models_for_league, predict_poisson_from_models
from src.models.train import train_league


def run_pipeline(round_number: int, data_path: str, league_name: str) -> pd.DataFrame:
    """
    Run end-to-end pipeline for a given league and round:
      1) Read raw match data for the selected league
      2) Preprocess and feature-engineer
      3) Filter upcoming matches for the given round
      4) Load pretrained models & scalers, training if missing
      5) Predict outcome probabilities
    """
    # 1) Read raw data
    key = league_name.lower().replace(" ", "_")
    raw_file = os.path.join(data_path, "raw", f"{key}_matches_full.csv")
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Missing raw data file: {raw_file}")
    df_all = pd.read_csv(raw_file, parse_dates=["date"])

    # 2) Preprocess & feature-engineer
    stat_windows = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}
    df_processed = process_matches(df_all, stat_windows, league_name)

    # 3) Filter future matches for the selected round
    df_future = df_processed[
        (df_processed["round"] == round_number) & (df_processed["result_home"].isna())
    ]
    if df_future.empty:
        return pd.DataFrame()

    # 4) Build feature lists
    features_home = (
        [f"{stat}_home_roll{w}" for stat in ("xg", "gf") for w in stat_windows[stat]]
        + [f"xg_conceded_away_roll{w}" for w in stat_windows["xg"]]
        + [f"ga_away_roll{w}" for w in stat_windows["ga"]]
        + ["avg_goals_for_home", "avg_goals_against_away"]
    )
    features_away = (
        [f"{stat}_away_roll{w}" for stat in ("xg", "gf") for w in stat_windows[stat]]
        + [f"xg_conceded_home_roll{w}" for w in stat_windows["xg"]]
        + [f"ga_home_roll{w}" for w in stat_windows["ga"]]
        + ["avg_goals_for_away", "avg_goals_against_home"]
    )

    # 5) Load or train models
    models_dir = os.path.join(data_path, "models")
    try:
        model_home, model_away, scaler_home, scaler_away = load_models_for_league(
            league_name, models_dir=models_dir
        )
    except FileNotFoundError:
        print(f"[INFO] Models for '{league_name}' not found. Training now...")
        train_league(
            league_name=league_name,
            data_dir=data_path,
            models_dir=models_dir,
            features_home=features_home,
            features_away=features_away,
        )
        model_home, model_away, scaler_home, scaler_away = load_models_for_league(
            league_name, models_dir=models_dir
        )

    # 6) Predict probabilities
    result_df = predict_poisson_from_models(
        model_home=model_home,
        model_away=model_away,
        scaler_home=scaler_home,
        scaler_away=scaler_away,
        df_future=df_future,
        features_home=features_home,
        features_away=features_away,
        max_goals=10,
    )

    return result_df
