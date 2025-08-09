import pandas as pd
import numpy as np
import pytest

from src.features.features import (
    _compute_relegated_averages,
    calculate_team_form_features,
    calculate_conceded_form_features,
    calculate_static_features,
    add_all_features,
)

# --- Tests for _compute_relegated_averages ---


def test_compute_relegated_averages_single_spot():
    # One match: A wins (3 pts), B loses (0 pts)
    df = pd.DataFrame(
        {
            "season": ["2025-2026"],
            "home_team": ["A"],
            "away_team": ["B"],
            "result_home": [1],
        }
    )
    agg_prev = pd.DataFrame(
        [
            {
                "season": "2025-2026",
                "team": "A",
                "avg_goals_for_prev": 1.0,
                "avg_goals_against_prev": 2.0,
            },
            {
                "season": "2025-2026",
                "team": "B",
                "avg_goals_for_prev": 3.0,
                "avg_goals_against_prev": 4.0,
            },
        ]
    )
    # bottom team is B (0 points)
    out = _compute_relegated_averages(df, agg_prev, spots=1)
    assert "2025-2026" in out
    assert out["2025-2026"]["for"] == pytest.approx(3.0)
    assert out["2025-2026"]["against"] == pytest.approx(4.0)

    # spots=2: both teams; average of for=(1+3)/2=2.0, against=(2+4)/2=3.0
    out2 = _compute_relegated_averages(df, agg_prev, spots=2)
    assert out2["2025-2026"]["for"] == pytest.approx(2.0)
    assert out2["2025-2026"]["against"] == pytest.approx(3.0)


# --- Tests for calculate_team_form_features ---


def test_calculate_team_form_features_window1():
    # Two matches where A plays home then away, B plays away then home
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "xg_home": [2.0, 3.0],
            "xg_away": [1.0, 4.0],
        }
    )
    out = calculate_team_form_features(df, stats=["xg"], windows=[1])
    # For second match (index 1):
    # xg_away_roll1 of A should equal previous A xg (from first match home xg_home=2.0)
    assert out.loc[1, "xg_away_roll1"] == pytest.approx(2.0)
    # xg_home_roll1 of B should equal previous B xg (from first match away xg_away=1.0)
    assert out.loc[1, "xg_home_roll1"] == pytest.approx(1.0)
    # First match form should be NaN (no previous data)
    assert np.isnan(out.loc[0, "xg_home_roll1"])
    assert np.isnan(out.loc[0, "xg_away_roll1"])


# --- Tests for calculate_conceded_form_features ---


def test_calculate_conceded_form_features_window1():
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "xg_home": [2.0, 3.0],
            "xg_away": [1.0, 4.0],
        }
    )
    out = calculate_conceded_form_features(df, windows=[1])
    # On second match: B as home concedes previous B conceded from match1 away xg_home=2.0
    assert out.loc[1, "xg_conceded_home_roll1"] == pytest.approx(2.0)
    # A as away concedes previous A conceded from match1 home xg_away=1.0
    assert out.loc[1, "xg_conceded_away_roll1"] == pytest.approx(1.0)
    # First match should be NaN
    assert np.isnan(out.loc[0, "xg_conceded_home_roll1"])
    assert np.isnan(out.loc[0, "xg_conceded_away_roll1"])


# --- Tests for calculate_static_features ---


def test_calculate_static_features_basic():
    # Two home matches for A in same season
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "season": ["2025-2026", "2025-2026"],
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "gf_home": [2, 3],
            "ga_home": [1, 1],
            "gf_away": [0, 1],
            "ga_away": [2, 1],
            "result_home": [1, 1],  # home won both matches
        }
    )
    out = calculate_static_features(df, agg_window=1)
    # avg_goals_for_curr_home = mean of [2,3] = 2.5
    assert out["avg_goals_for_curr_home"].iloc[0] == pytest.approx(2.5)
    # avg_goals_for_curr_away = mean of [0,1] = 0.5
    assert out["avg_goals_for_curr_away"].iloc[1] == pytest.approx(0.5)
    # matches_played_home: 0 then 1
    assert out["matches_played_home"].tolist() == [0, 1]
    # matches_played_away: 0 then 1
    assert out["matches_played_away"].tolist() == [0, 1]


# --- Tests for add_all_features ---


def test_add_all_features_combines_all(tmp_path):
    # Simple two-match df
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "season": ["2025-2026", "2025-2026"],
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "xg_home": [2.0, 3.0],
            "xg_away": [1.0, 4.0],
            "gf_home": [2, 3],
            "ga_home": [1, 1],
            "gf_away": [0, 1],
            "ga_away": [2, 1],
            "result_home": [1, 1],  # both home wins
        }
    )
    stat_windows = {"xg": [1], "gf": [1], "ga": [1]}
    out = add_all_features(df, stat_windows, agg_window=1)
    # Should contain form, conceded, static and match_played columns
    assert "xg_home_roll1" in out.columns
    assert "xg_conceded_home_roll1" in out.columns
    assert "avg_goals_for_curr_home" in out.columns
    assert "matches_played_home" in out.columns
