# File: tests/test_odds.py
import math
import numpy as np
import pandas as pd
import pytest

# Importer funksjoner vi tester
from src.models.odds import (
    _get_lambdas,
    calculate_hub_odds,
    calculate_btts_odds,
    calculate_over_under_odds,
)

# Vi bruker compute_match_outcome_probabilities fra predict-modulen i odds,
# men lambdas (modell-output) mockes gjennom load_models_for_league.

# ---------- Hjelpe-fixtures ----------


@pytest.fixture
def minimal_features():
    """
    Samme struktur som i treningen (se update_all.py):
    features_home:
      xg_home_roll{5,10}, gf_home_roll{5,10}, xg_conceded_away_roll{5,10}, ga_away_roll{5,10},
      avg_goals_for_home, avg_goals_against_away
    features_away:
      xg_away_roll{5,10}, gf_away_roll{5,10}, xg_conceded_home_roll{5,10}, ga_home_roll{5,10},
      avg_goals_for_away, avg_goals_against_home
    """
    features_home = [
        "xg_home_roll5",
        "xg_home_roll10",
        "gf_home_roll5",
        "gf_home_roll10",
        "xg_conceded_away_roll5",
        "xg_conceded_away_roll10",
        "ga_away_roll5",
        "ga_away_roll10",
        "avg_goals_for_home",
        "avg_goals_against_away",
    ]
    features_away = [
        "xg_away_roll5",
        "xg_away_roll10",
        "gf_away_roll5",
        "gf_away_roll10",
        "xg_conceded_home_roll5",
        "xg_conceded_home_roll10",
        "ga_home_roll5",
        "ga_home_roll10",
        "avg_goals_for_away",
        "avg_goals_against_home",
    ]
    return features_home, features_away


@pytest.fixture
def minimal_df(minimal_features):
    """
    Ett-kamps DataFrame med alle nødvendige kolonner for odds-funksjonene.
    Inkluderer date/time + teamnavn (trengs for team-dummies).
    """
    features_home, features_away = minimal_features
    data = {
        "date": [pd.Timestamp("2025-08-21")],
        "time": ["18:00"],
        "home_team": ["Team A"],
        "away_team": ["Team B"],
        # Home features
        "xg_home_roll5": [1.4],
        "xg_home_roll10": [1.3],
        "gf_home_roll5": [1.6],
        "gf_home_roll10": [1.4],
        "xg_conceded_away_roll5": [1.1],
        "xg_conceded_away_roll10": [1.2],
        "ga_away_roll5": [1.0],
        "ga_away_roll10": [1.1],
        "avg_goals_for_home": [1.5],
        "avg_goals_against_away": [1.2],
        # Away features
        "xg_away_roll5": [1.2],
        "xg_away_roll10": [1.1],
        "gf_away_roll5": [1.3],
        "gf_away_roll10": [1.2],
        "xg_conceded_home_roll5": [1.3],
        "xg_conceded_home_roll10": [1.4],
        "ga_home_roll5": [1.2],
        "ga_home_roll10": [1.3],
        "avg_goals_for_away": [1.1],
        "avg_goals_against_home": [1.4],
    }
    df = pd.DataFrame(data)
    # Sikkerhet: sørg for at alle features faktisk finnes som kolonner
    for col in set(minimal_features[0] + minimal_features[1]):
        assert col in df.columns
    return df


class FakeScaler:
    """
    En enkel 'scaler' som bare returnerer input uendret.
    Vi trenger bare feature_names_in_ for reindex i _get_lambdas.
    """

    def __init__(self, feature_names_in_):
        self.feature_names_in_ = np.array(feature_names_in_)
        self.last_columns = None

    def transform(self, X):
        # Lagre kolonnerekkefølge for inspeksjon
        if hasattr(X, "columns"):
            self.last_columns = list(X.columns)
            # Returner som numpy-array i riktig rekkefølge
            return X[self.feature_names_in_].to_numpy()
        # Om noen sender array direkte:
        return X


class FakeModel:
    """
    En veldig enkel 'modell' som returnerer faste lambdas for to rader (hjemme/borte).
    """

    def predict(self, X):
        # For én kamp har vi 2 rader (Xh, Xa). Returner to verdier.
        n = len(X)
        assert n in (2,), "Forventet 2 rader (Xh, Xa) for én kamp"
        # Harde, deterministiske lamdba-verdier:
        return np.array([1.50, 1.00])


def build_feature_space_for_scaler(df, features_home, features_away):
    """
    Reproduserer kolonnenavnene som _get_lambdas vil bygge opp etter stripping og dummies.
    - Felles feature-navn etter stripping av _home/_away
    - is_home
    - team-dummies for Team A og Team B, både att_ og def_
    Legg gjerne på noen 'ekstra' features for å teste reindex-fylling med 0.
    """
    # Strip fra hjemmelag
    stripped_home = [c.replace("_home", "").replace("_away", "") for c in features_home]
    stripped_away = [c.replace("_home", "").replace("_away", "") for c in features_away]
    # Felles numeriske features (samme sett for home og away etter stripping)
    base_feats = sorted(set(stripped_home + stripped_away))

    # Team-dummies som _add_team_dummies vil lage for (Team A vs Team B)
    team_dummy_cols = ["att_Team A", "att_Team B", "def_Team A", "def_Team B"]

    # Til slutt is_home-flagget
    all_feats = base_feats + ["is_home"] + team_dummy_cols

    # La oss også legge inn noen ekstra features som ikke finnes i X_all
    # for å verifisere at reindex fyller dem med 0 uten å knekke.
    extras = ["att_Team C", "def_Team C", "random_unused_feature"]
    return all_feats + extras


@pytest.fixture
def patched_models(monkeypatch, minimal_df, minimal_features):
    """
    Monkeypatch load_models_for_league til å returnere FakeModel + FakeScaler
    med passende feature_names_in_.
    """
    df = minimal_df
    features_home, features_away = minimal_features
    feature_space = build_feature_space_for_scaler(df, features_home, features_away)
    scaler = FakeScaler(feature_space)
    model = FakeModel()

    def fake_loader(league, models_dir):
        return model, scaler

    # Patch inn loaderen
    monkeypatch.setattr(
        "src.models.odds.load_models_for_league",
        fake_loader,
        raising=True,
    )

    # Returner også ref til scaler og model så testene kan inspisere
    return {"scaler": scaler, "model": model}


# ---------- Tester for _get_lambdas ----------


def test_get_lambdas_returns_two_values(minimal_df, minimal_features, patched_models):
    df = minimal_df
    features_home, features_away = minimal_features

    lam_h, lam_a = _get_lambdas(
        df=df,
        features_home=features_home,
        features_away=features_away,
        league="Premier League",
        models_dir="data/models",
    )
    assert pytest.approx(lam_h, rel=1e-6) == 1.50
    assert pytest.approx(lam_a, rel=1e-6) == 1.00


def test_get_lambdas_reindexes_with_missing_columns(
    minimal_df, minimal_features, patched_models
):
    """
    Sikrer at reindex fungerer selv om scaler.feature_names_in_ inneholder ekstra kolonner
    som ikke er i X_all (skal fylles med 0).
    """
    df = minimal_df
    features_home, features_away = minimal_features

    lam_h, lam_a = _get_lambdas(
        df=df,
        features_home=features_home,
        features_away=features_away,
        league="Premier League",
        models_dir="data/models",
    )
    assert lam_h > 0 and lam_a > 0  # sanity
    # Sjekk at scaler fikk kolonner i nøyaktig samme rekkefølge som feature_names_in_
    scaler = patched_models["scaler"]
    assert scaler.last_columns == list(scaler.feature_names_in_)


# ---------- Tester for calculate_hub_odds ----------


def parse_percent_string(s):
    # "64.3%" -> 0.643
    assert s.endswith("%")
    return float(s[:-1]) / 100.0


def test_calculate_hub_odds_output_format(minimal_df, minimal_features, patched_models):
    df = minimal_df
    features_home, features_away = minimal_features

    out = calculate_hub_odds(
        df=df,
        features_home=features_home,
        features_away=features_away,
        league="Premier League",
        models_dir="data/models",
        max_goals=10,
    )

    # 3 rader: Hjemme, Uavgjort, Borte
    assert list(out["Utfall"]) == ["Hjemmeseier", "Uavgjort", "Borteseier"]

    # Kolonner og format
    assert set(out.columns) == {"Utfall", "Sannsynlighet", "Fair odds"}
    # Sannsynlighet: én desimal + '%' (f.eks. '54.2%')
    for p in out["Sannsynlighet"]:
        assert isinstance(p, str)
        assert p.endswith("%")
        # sjekk at det er én desimal
        whole, pct = p[:-1].split(".")
        assert len(pct) == 1

    # Fair odds: to desimaler
    for o in out["Fair odds"]:
        assert isinstance(o, str)
        whole, dec = o.split(".")
        assert len(dec) == 2

    # Sannsynligheter bør summeres ~1.0 (numerisk, ikke formatert)
    probs = [parse_percent_string(p) for p in out["Sannsynlighet"]]
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-3, abs_tol=1e-3)


# ---------- Tester for calculate_btts_odds ----------


def test_calculate_btts_odds(minimal_df, minimal_features, patched_models):
    df = minimal_df
    features_home, features_away = minimal_features

    out = calculate_btts_odds(
        df=df,
        features_home=features_home,
        features_away=features_away,
        league="Premier League",
        models_dir="data/models",
    )

    assert list(out["Utfall"]) == ["Begge lag scorer – Ja", "Begge lag scorer – Nei"]
    # Gyldige prosenter og odds
    for p in out["Sannsynlighet"]:
        v = parse_percent_string(p)
        assert 0.0 <= v <= 1.0
    for o in out["Fair odds"]:
        v = float(o)
        assert v > 1.0  # fair odds er > 1 for sannsynlighet < 1

    # Ja + Nei ~ 1
    probs = [parse_percent_string(p) for p in out["Sannsynlighet"]]
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-3, abs_tol=1e-3)


# ---------- Tester for calculate_over_under_odds ----------


@pytest.mark.parametrize("threshold", [1.5, 2.5, 3.5])
def test_calculate_over_under_odds(
    minimal_df, minimal_features, patched_models, threshold
):
    df = minimal_df
    features_home, features_away = minimal_features

    out = calculate_over_under_odds(
        df=df,
        features_home=features_home,
        features_away=features_away,
        league="Premier League",
        models_dir="data/models",
        threshold=threshold,
    )

    assert list(out["Utfall"]) == [f"Under {threshold}", f"Over {threshold}"]
    probs = [parse_percent_string(p) for p in out["Sannsynlighet"]]
    # Sum ~ 1
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    # Odds-format (to desimaler)
    for o in out["Fair odds"]:
        whole, dec = o.split(".")
        assert len(dec) == 2
