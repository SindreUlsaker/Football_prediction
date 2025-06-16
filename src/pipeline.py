import pandas as pd
from src.data.process import process_matches
from src.models.train import train_poisson_models
from src.models.predict import predict_poisson_from_models

def run_pipeline(round_number: int, data_path: str) -> pd.DataFrame:
    # 1) Les inn rådata
    df_all = pd.read_csv(
        f"{data_path}/raw/premier_league_matches_full.csv",
        parse_dates=["date"]
    )

    # 2) Definer rolling-statistikkvinduer
    stat_windows = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}

    # 3) Rens og lag features
    processed = process_matches(df_all, stat_windows)

    # 4) Bygg feature-lister
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

    # 5) Tren modeller
    model_home, model_away, scaler_home, scaler_away = train_poisson_models(
        processed, features_home, features_away
    )

    # 6) Filtrer kamper for valgt runde
    df_future = processed[
        (processed["round"] == round_number) & (processed["result_home"].isna())
    ].copy()
    if df_future.empty:
        return pd.DataFrame()

    # 7) Generer sannsynligheter
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