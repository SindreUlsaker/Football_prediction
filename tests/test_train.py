import os
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from src.models.train import train_poisson_model, _add_team_dummies, train_league

# --- Tests for _add_team_dummies ---


def test_add_team_dummies_alignment():
    df_home = pd.DataFrame(
        {
            "home_team": ["A", "B"],
            "away_team": ["X", "Y"],
        }
    )
    df_away = pd.DataFrame(
        {
            "home_team": ["A", "B"],
            "away_team": ["X", "Y"],
        }
    )
    Xh, Xa = _add_team_dummies(df_home, df_away)
    # Expected dummy columns for all teams (home and away)
    teams = set(df_home["home_team"]) | set(
        df_home["away_team"]
    )  # {'A', 'B', 'X', 'Y'}
    expected = {f"att_{t}" for t in teams} | {f"def_{t}" for t in teams}
    assert set(Xh.columns) == expected
    assert set(Xa.columns) == expected
    # Values for home-team dummies in Xh
    assert Xh.loc[0, "att_A"] == 1
    assert Xh.loc[0, "def_X"] == 1
    assert all(
        Xh.loc[
            0,
            [f"att_{t}" for t in teams if t not in ["A"]]
            + [f"def_{t}" for t in teams if t not in ["X"]],
        ]
        == 0
    )
    # Values for away-team dummies in Xa
    assert Xa.loc[0, "att_X"] == 1
    assert Xa.loc[0, "def_A"] == 1
    assert all(
        Xa.loc[
            0,
            [f"att_{t}" for t in teams if t not in ["X"]]
            + [f"def_{t}" for t in teams if t not in ["A"]],
        ]
        == 0
    )


# --- Tests for train_poisson_model ---


def test_train_poisson_model_basic():
    # Synthetic dataset with two matches
    data = pd.DataFrame(
        {
            "home_team": ["A", "B"],
            "away_team": ["X", "Y"],
            "gf_home": [1, 2],
            "gf_away": [0, 1],
            "xg_home": [1.0, 0.8],
            "ga_home": [0.5, 1.2],
            "xg_away": [0.7, 0.6],
            "ga_away": [1.5, 0.9],
        }
    )
    features_home = ["xg_home", "ga_home"]
    features_away = ["xg_away", "ga_away"]
    model, scaler = train_poisson_model(data, features_home, features_away)
    # Check types
    assert isinstance(model, PoissonRegressor)
    assert isinstance(scaler, StandardScaler)
    # Feature names and model coefficients align
    n_feats = len(scaler.feature_names_in_)
    assert model.coef_.shape[0] == n_feats
    # Model should predict non-negative lambdas for zero input
    X_zero = np.zeros((1, n_feats))
    lam = model.predict(scaler.transform(X_zero))
    assert lam.shape == (1,)
    assert lam[0] >= 0


# --- Tests for train_league ---


def test_train_league_file_not_found(tmp_path):
    # Expect error if processed file missing
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    with pytest.raises(FileNotFoundError):
        train_league(
            league_name="TEST",
            data_dir=str(data_dir),
            models_dir=str(models_dir),
            features_home=["x"],
            features_away=["y"],
        )


def test_train_league_saves_model_and_scaler(tmp_path):
    # Setup temporary processed CSV
    data_dir = tmp_path / "data"
    proc_dir = data_dir / "processed"
    proc_dir.mkdir(parents=True)
    key = "test"
    processed_file = proc_dir / f"{key}_processed.csv"
    # Create minimal processed DataFrame
    df = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "gf_home": [1],
            "gf_away": [0],
            "xg_home": [1.0],
            "ga_home": [0.5],
            "xg_away": [0.7],
            "ga_away": [1.2],
            "home_team": ["A"],
            "away_team": ["X"],
        }
    )
    df.to_csv(processed_file, index=False)

    models_dir = tmp_path / "models"
    # Call train_league
    train_league(
        league_name="TEST",
        data_dir=str(data_dir),
        models_dir=str(models_dir),
        features_home=["xg_home", "ga_home"],
        features_away=["xg_away", "ga_away"],
    )
    # Assert files exist
    model_path = models_dir / f"{key}_model.joblib"
    scaler_path = models_dir / f"{key}_scaler.joblib"
    assert model_path.exists()
    assert scaler_path.exists()
    # Load and check types
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    assert isinstance(loaded_model, PoissonRegressor)
    assert isinstance(loaded_scaler, StandardScaler)
