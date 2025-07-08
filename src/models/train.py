from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import joblib


def train_poisson_model(
    data: pd.DataFrame, features_home: list[str], features_away: list[str]
) -> tuple[PoissonRegressor, StandardScaler]:
    """
    Train a single PoissonRegressor on both home and away goals,
    using an `is_home` feature to distinguish home/away.

    Parameters:
      - data: processed DataFrame with columns for home/away stats and targets
      - features_home: list of column names for home features (e.g. 'xg_home', 'gf_home')
      - features_away: list of column names for away features (e.g. 'xg_away', 'gf_away')

    Returns:
      - Trained PoissonRegressor
      - Fitted StandardScaler
    """
    # Only drop rows where we know the goal outcome
    df_home = data[data["gf_home"].notna()].copy()
    df_away = data[data["gf_away"].notna()].copy()

    # Home-team features
    Xh = df_home[features_home].copy()
    Xh.columns = [c.replace("_home", "").replace("_away", "") for c in Xh.columns]
    Xh["is_home"] = 1
    Xh = Xh.fillna(0)
    yh = df_home["gf_home"]

    # Away-team features
    Xa = df_away[features_away].copy()
    Xa.columns = [c.replace("_home", "").replace("_away", "") for c in Xa.columns]
    Xa["is_home"] = 0
    Xa = Xa.fillna(0)
    ya = df_away["gf_away"]

    # Legg på team-fixed-effects
    dum_h, dum_a = _add_team_dummies(df_home, df_away)
    Xh = pd.concat([Xh, dum_h], axis=1)
    Xa = pd.concat([Xa, dum_a], axis=1)

    # Combine both perspectives
    X_all = pd.concat([Xh, Xa], ignore_index=True).fillna(0)
    y_all = pd.concat([yh, ya], ignore_index=True)

    # Scale features
    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all)

    # Train Poisson regressor
    model = PoissonRegressor(alpha=1.0, max_iter=300).fit(X_scaled, y_all)
    return model, scaler


def _add_team_dummies(df_home, df_away):
    """
    Lager attack- og defence-dummies for hvert lag.
    Returnerer to DataFrames (Xh, Xa) med like kolonner.
    """
    # Angreps-dummies: hvem som skyter
    att_home = pd.get_dummies(df_home["home_team"], prefix="att")
    att_away = pd.get_dummies(df_away["away_team"], prefix="att")
    # Forsvars-dummies: hvem som forsvarer
    def_home = pd.get_dummies(df_home["away_team"], prefix="def")
    def_away = pd.get_dummies(df_away["home_team"], prefix="def")
    # Sett sammen
    Xh = pd.concat([att_home, def_home], axis=1)
    Xa = pd.concat([att_away, def_away], axis=1)
    # Sørg for at de har samme kolonner (utfyll med 0 der det mangler)
    return Xh.align(Xa, join="outer", axis=1, fill_value=0)


def train_league(
    league_name: str,
    data_dir: str,
    models_dir: str,
    features_home: list[str],
    features_away: list[str],
) -> None:
    """
    Read processed data for the given league, train a single Poisson model,
    and save both model and scaler to disk.
    """
    key = league_name.lower().replace(" ", "_")
    processed_file = os.path.join(data_dir, "processed", f"{key}_processed.csv")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed data not found for league: {league_name}")

    df = pd.read_csv(processed_file, parse_dates=["date"])

    # Train model and scaler
    model, scaler = train_poisson_model(df, features_home, features_away)

    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Save model and scaler
    joblib.dump(model, os.path.join(models_dir, f"{key}_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, f"{key}_scaler.joblib"))

    print(f"[INFO] Trained and saved model for {league_name}")
