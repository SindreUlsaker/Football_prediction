from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import joblib


def train_poisson_models(
    data: pd.DataFrame, features_home: list[str], features_away: list[str]
):
    """
    Trener Poisson-modeller for gf_home og gf_away med egne scalers.
    """
    df = data.dropna(subset=features_home + features_away + ["gf_home", "gf_away"])

    # Hjemmelag
    Xh = df[features_home].copy()
    Xh["is_home"] = 1
    yh = df["gf_home"]

    # Bortelag
    Xa = df[features_away].copy()
    Xa["is_home"] = 0
    ya = df["gf_away"]

    scaler_home = StandardScaler().fit(Xh)
    scaler_away = StandardScaler().fit(Xa)

    Xh_s = scaler_home.transform(Xh)
    Xa_s = scaler_away.transform(Xa)

    model_home = PoissonRegressor(alpha=1.0, max_iter=300).fit(Xh_s, yh)
    model_away = PoissonRegressor(alpha=1.0, max_iter=300).fit(Xa_s, ya)

    return model_home, model_away, scaler_home, scaler_away


def train_league(
    league_name: str,
    data_dir: str,
    models_dir: str,
    features_home: list[str],
    features_away: list[str],
):
    """
    Leser processed data for gitt liga, trener modeller og lagrer dem.
    """
    key = league_name.lower().replace(" ", "_")
    processed_file = os.path.join(data_dir, "processed", f"{key}_processed.csv")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed data not found for league: {league_name}")

    df = pd.read_csv(processed_file, parse_dates=["date"])

    model_home, model_away, scaler_home, scaler_away = train_poisson_models(
        df, features_home, features_away
    )

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model_home, os.path.join(models_dir, f"{key}_model_home.joblib"))
    joblib.dump(model_away, os.path.join(models_dir, f"{key}_model_away.joblib"))
    joblib.dump(scaler_home, os.path.join(models_dir, f"{key}_scaler_home.joblib"))
    joblib.dump(scaler_away, os.path.join(models_dir, f"{key}_scaler_away.joblib"))

    print(f"[INFO] Trente og lagret modeller for {league_name}")
