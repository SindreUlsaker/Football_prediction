import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson


def load_models_for_league(league_name: str, models_dir: str = "models") -> tuple:
    """
    Load Poisson models and scalers for given league from disk.

    Expects files in models_dir:
      {league_key}_model_home.joblib
      {league_key}_model_away.joblib
      {league_key}_scaler_home.joblib
      {league_key}_scaler_away.joblib
    Returns:
      (model_home, model_away, scaler_home, scaler_away)
    """
    key = league_name.lower().replace(" ", "_")
    paths = {
        "model_home": os.path.join(models_dir, f"{key}_model_home.joblib"),
        "model_away": os.path.join(models_dir, f"{key}_model_away.joblib"),
        "scaler_home": os.path.join(models_dir, f"{key}_scaler_home.joblib"),
        "scaler_away": os.path.join(models_dir, f"{key}_scaler_away.joblib"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} at {path}")
    model_home = joblib.load(paths["model_home"])
    model_away = joblib.load(paths["model_away"])
    scaler_home = joblib.load(paths["scaler_home"])
    scaler_away = joblib.load(paths["scaler_away"])
    return model_home, model_away, scaler_home, scaler_away


def compute_match_outcome_probabilities(lambda_home, lambda_away, max_goals: int = 10):
    """
    Given expected goals for home and away, compute probabilities of
    home win, draw, and away win using Poisson distributions.
    """
    probs_home = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    probs_away = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    match_matrix = np.outer(probs_home, probs_away)

    p_home_win = np.tril(match_matrix, k=-1).sum()
    p_draw = np.trace(match_matrix)
    p_away_win = np.triu(match_matrix, k=1).sum()

    return round(p_home_win, 3), round(p_draw, 3), round(p_away_win, 3)


def predict_poisson_from_models(
    model_home,
    model_away,
    scaler_home,
    scaler_away,
    df_future: pd.DataFrame,
    features_home: list[str],
    features_away: list[str],
    max_goals: int = 10,
) -> pd.DataFrame:
    """
    Predict match outcome probabilities using two Poisson models.

    Parameters:
      - model_home: trained PoissonRegressor for home goals
      - model_away: trained PoissonRegressor for away goals
      - scaler_home: StandardScaler fitted on home features
      - scaler_away: StandardScaler fitted on away features
      - df_future: DataFrame of upcoming matches
      - features_home: list of column names for home model
      - features_away: list of column names for away model
      - max_goals: maximum goals to consider (default 10)

    Returns:
      DataFrame with columns date, home_team, away_team,
      lambda_home, lambda_away, prob_home, prob_draw, prob_away
    """
    # Ensure df_future has zero-based consecutive indices so that
    # lambda_home[i] and lambda_away[i] align with df_future.iloc[i]
    df_future = df_future.reset_index(drop=True)

    # Home features
    Xh = df_future[features_home].copy()
    Xh["is_home"] = 1
    Xh_scaled = scaler_home.transform(Xh)
    lambda_home = model_home.predict(Xh_scaled)

    # Away features
    Xa = df_future[features_away].copy()
    Xa["is_home"] = 0
    Xa_scaled = scaler_away.transform(Xa)
    lambda_away = model_away.predict(Xa_scaled)

    # Build output
    records = []
    for i, row in df_future.iterrows():
        lam_h = lambda_home[i]
        lam_a = lambda_away[i]
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
            }
        )
    return pd.DataFrame(records)
