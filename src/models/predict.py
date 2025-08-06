import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

def _add_team_dummies(df_home, df_away):
    att_home = pd.get_dummies(df_home["home_team"], prefix="att")
    att_away = pd.get_dummies(df_away["away_team"], prefix="att")
    def_home = pd.get_dummies(df_home["away_team"], prefix="def")
    def_away = pd.get_dummies(df_away["home_team"], prefix="def")
    Xh = pd.concat([att_home, def_home], axis=1)
    Xa = pd.concat([att_away, def_away], axis=1)
    return Xh.align(Xa, join="outer", axis=1, fill_value=0)

def load_models_for_league(league_name: str, models_dir: str = "models") -> tuple:
    """
    Load a single Poisson model and scaler for a league.
    """
    key = league_name.lower().replace(" ", "_")
    model_path = os.path.join(models_dir, f"{key}_model.joblib")
    scaler_path = os.path.join(models_dir, f"{key}_scaler.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found for league: {league_name}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def compute_match_outcome_probabilities(
    lam_h: float, lam_a: float, max_goals: int = 10
) -> tuple[float, float, float]:
    """
    Compute probabilities of home win, draw, and away win from Poisson lambdas.
    """
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i, j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
    prob_home = np.tril(prob_matrix, -1).sum()
    prob_draw = np.trace(prob_matrix)
    prob_away = np.triu(prob_matrix, 1).sum()
    return prob_home, prob_draw, prob_away


def predict_poisson_from_models(
    df: pd.DataFrame,
    features_home: list[str],
    features_away: list[str],
    league_name: str,
    models_dir: str = "models",
    max_goals: int = 10,
) -> pd.DataFrame:
    """
    Predict match outcome probabilities using a single Poisson model.

    Parameters:
      - df: DataFrame with upcoming matches and home/away features
      - features_home: list of column names ending with '_home'
      - features_away: list of column names ending with '_away'
      - league_name: league identifier for loading the model
      - models_dir: directory with saved models
      - max_goals: max goals to consider for Poisson

    Returns:
      - DataFrame with date, teams, lambdas, and win/draw probabilities
    """
    model, scaler = load_models_for_league(league_name, models_dir)

    df = df.reset_index(drop=True)

    # Prepare home-team inputs
    Xh = df[features_home].copy()
    Xh.columns = [c.replace("_home", "").replace("_away", "") for c in Xh.columns]
    Xh["is_home"] = 1
    Xh = Xh.fillna(0)

    # Prepare away-team inputs
    Xa = df[features_away].copy()
    Xa.columns = [c.replace("_home", "").replace("_away", "") for c in Xa.columns]
    Xa["is_home"] = 0
    Xa = Xa.fillna(0)
    
    dum_h, dum_a = _add_team_dummies(df, df)
    Xh = pd.concat([Xh, dum_h], axis=1)
    Xa = pd.concat([Xa, dum_a], axis=1)

    # Combine and scale
    X_all = pd.concat([Xh, Xa], ignore_index=True)
    X_all = X_all.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_scaled = scaler.transform(X_all)

    # Predict lambdas
    lambdas = model.predict(X_scaled)
    lambda_home = lambdas[: len(df)]
    lambda_away = lambdas[len(df) :]

    # Build results
    records = []
    for idx, row in df.iterrows():
        alpha = 0.3
        ratio = lambda_home[idx] / lambda_away[idx]
        X = ratio ** alpha
        Y = (1 / ratio) ** alpha
        lam_h = lambda_home[idx] * X
        lam_a = lambda_away[idx] * Y
        p_h, p_d, p_a = compute_match_outcome_probabilities(lam_h, lam_a, max_goals)
        records.append(
            {
                "date": row["date"],
                "time": row["time"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "prob_home": p_h,
                "prob_draw": p_d,
                "prob_away": p_a,
                "lambda_home": lambda_home[idx],
                "lambda_away": lambda_away[idx],
            }
        )
    return pd.DataFrame(records)
