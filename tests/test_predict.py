# File: tests/test_predict.py
import os
import math
import joblib
import numpy as np
import pandas as pd
import pytest

# Importer funksjoner fra predict.py
from src.models.predict import (
    _add_team_dummies,
    load_models_for_league,
    compute_match_outcome_probabilities,
    predict_poisson_from_models,
)

# ----------------------------
# Hjelpeklasser for modell/scaler
# ----------------------------


class DummyScaler:
    """
    Identitetsscaler som beholder DataFrame uendret.
    Må ha feature_names_in_ og transform().
    """

    def __init__(self, feature_names):
        # predict.py forventer at vi kan gjøre reindex(columns=scaler.feature_names_in_)
        self.feature_names_in_ = list(feature_names)

    def transform(self, X):
        # returner DataFrame intakt (predict() i DummyModel forventer å se kolonnenavn)
        return X


class DummyModel:
    """
    Enkel modell der prediksjon kun avhenger av is_home:
    lambda = base + bump * is_home

    Når predict_poisson_from_models bygger X_all = [Xh; Xa],
    vil de første N radene ha is_home=1 (hjemme), og de neste N har is_home=0 (borte).
    """

    def __init__(self, base=1.0, bump=0.2):
        self.base = float(base)
        self.bump = float(bump)

    def predict(self, X):
        # X er en DataFrame fordi DummyScaler.transform returnerer DataFrame
        is_home = X["is_home"].to_numpy().astype(float)
        return self.base + self.bump * is_home


# ----------------------------
# Pytest fixtures
# ----------------------------


@pytest.fixture
def league_name():
    return "Premier League"  # key -> 'premier_league'


@pytest.fixture
def models_dir(tmp_path, league_name):
    """
    Oppretter en midlertidig models-dir med dummy scaler og modell dumpet med joblib.
    """
    mdir = tmp_path / "models"
    mdir.mkdir(parents=True, exist_ok=True)

    key = league_name.lower().replace(" ", "_")
    model_path = mdir / f"{key}_model.joblib"
    scaler_path = mdir / f"{key}_scaler.joblib"

    # Sett opp kolonnenavnene slik predict.py forventer dem etter prosessering:
    # - features (uten _home/_away)
    # - is_home
    # - team-dummies (att_* og def_*)
    feature_names = [
        "f1",
        "f2",
        "is_home",
        "att_Team A",
        "def_Team B",
        "att_Team B",
        "def_Team A",
    ]

    joblib.dump(DummyModel(base=1.0, bump=0.2), model_path)
    joblib.dump(DummyScaler(feature_names), scaler_path)

    return str(mdir)


@pytest.fixture
def minimal_df():
    """
    Minimal DataFrame for to kamper, med nødvendige kolonner:
    date, time, home_team, away_team + featurekolonner.
    """
    data = [
        {
            "date": pd.Timestamp("2025-08-20"),
            "time": "16:00",
            "home_team": "Team A",
            "away_team": "Team B",
            "f1_home": 0.6,
            "f2_home": 0.4,
            "f1_away": 0.5,
            "f2_away": 0.3,
        },
        {
            "date": pd.Timestamp("2025-08-21"),
            "time": "18:00",
            "home_team": "Team B",
            "away_team": "Team A",
            "f1_home": 0.7,
            "f2_home": 0.2,
            "f1_away": 0.4,
            "f2_away": 0.6,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def features_home():
    # Kolonner som ender på _home
    return ["f1_home", "f2_home"]


@pytest.fixture
def features_away():
    # Kolonner som ender på _away
    return ["f1_away", "f2_away"]


# ----------------------------
# Tester
# ----------------------------


def test_compute_match_outcome_probabilities_basic_properties():
    # Enkle sanity checks
    lam_h, lam_a = 1.5, 1.0

    # Øk max_goals for mindre truncation og tillat litt underskudd
    p_h, p_d, p_a = compute_match_outcome_probabilities(lam_h, lam_a, max_goals=20)

    assert 0 <= p_h <= 1
    assert 0 <= p_d <= 1
    assert 0 <= p_a <= 1

    s = float(p_h + p_d + p_a)
    # Ingen inflasjon, og aksepter maks 1e-5 i manglende masse pga truncation
    assert s <= 1.0
    assert (1.0 - s) < 1e-5


def test_add_team_dummies_alignment(minimal_df):
    # For én rad skal vi få att_homeTeam/def_awayTeam for home,
    # og att_awayTeam/def_homeTeam for away, og align’e kolonner.
    df = minimal_df.iloc[[0]].copy()
    Xh, Xa = _add_team_dummies(df, df)

    # Kolonnenavnene må matche og være samme rekkefølge etter align
    assert list(Xh.columns) == list(Xa.columns)

    # Sjekk at forventede dummies finnes
    expected = {"att_Team A", "def_Team B", "att_Team B", "def_Team A"}
    assert expected.issubset(set(Xh.columns))

    # Hjemme/away radene bør ha forskjellig aktivering av dummies
    # (ikke krav om eksakte verdier her, men bør ikke være alle 0)
    assert Xh.to_numpy().sum() > 0
    assert Xa.to_numpy().sum() > 0


def test_load_models_for_league_missing(tmp_path, league_name):
    # Ingen filer lagt inn -> forvent FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_models_for_league(league_name, models_dir=str(tmp_path))


def test_predict_poisson_from_models_single_row(
    models_dir, league_name, features_home, features_away
):
    # Én kamp
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-08-20"),
                "time": "16:00",
                "home_team": "Team A",
                "away_team": "Team B",
                "f1_home": 0.6,
                "f2_home": 0.4,
                "f1_away": 0.5,
                "f2_away": 0.3,
            }
        ]
    )

    out = predict_poisson_from_models(
        df=df,
        features_home=features_home,
        features_away=features_away,
        league_name=league_name,
        models_dir=models_dir,
        max_goals=20,  # høyere for å redusere truncation-tap
    )

    # Forvent ett resultat med korrekte kolonner
    expected_cols = {
        "date",
        "time",
        "home_team",
        "away_team",
        "prob_home",
        "prob_draw",
        "prob_away",
        "lambda_home",
        "lambda_away",
    }
    assert expected_cols.issubset(set(out.columns))
    assert len(out) == 1

    # Sannsynligheter i [0,1] og summerer til ~1 (tillat bittelitt underskudd)
    p_h, p_d, p_a = out.loc[0, ["prob_home", "prob_draw", "prob_away"]]
    assert 0 <= p_h <= 1 and 0 <= p_d <= 1 and 0 <= p_a <= 1

    s = float(p_h + p_d + p_a)
    assert s <= 1.0
    assert (1.0 - s) < 1e-5

    # DummyModel gir høyere lambda for hjemme (is_home=1) enn borte
    lam_h = float(out.loc[0, "lambda_home"])
    lam_a = float(out.loc[0, "lambda_away"])
    assert lam_h > lam_a

    # Det bør reflekteres i høyere hjemmeseier-sannsynlighet enn borteseier
    assert p_h > p_a


def test_predict_poisson_from_models_two_rows(
    models_dir, league_name, minimal_df, features_home, features_away
):
    # To kamper på én gang
    out = predict_poisson_from_models(
        df=minimal_df,
        features_home=features_home,
        features_away=features_away,
        league_name=league_name,
        models_dir=models_dir,
        max_goals=20,  # øk for konsistente summer
    )

    assert len(out) == 2
    for _, row in out.iterrows():
        s = float(row["prob_home"] + row["prob_draw"] + row["prob_away"])
        # numerisk robusthet, tillat bittelitt underskudd
        assert s <= 1.0
        assert (1.0 - s) < 1e-5
        assert 0 <= row["prob_home"] <= 1
        assert 0 <= row["prob_draw"] <= 1
        assert 0 <= row["prob_away"] <= 1
        assert row["lambda_home"] > 0
        assert row["lambda_away"] > 0
