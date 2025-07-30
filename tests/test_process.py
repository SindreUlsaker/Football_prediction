import os
import pandas as pd
import pytest
from src.data.process import preprocess_data, ensure_numeric, process_matches

# --- Fixtures and helpers ---
@pytest.fixture
def sample_raw_df():
    # Raw data with a home match, an away match, and an invalid row
    data = {
        "date": ["2025-07-29", "2025-07-30", None],
        "time": ["18:00", "20:00", "19:00"],
        "comp": ["TEST", "TEST", "TEST"],
        "season": ["2025-2026", "2025-2026", None],
        "team": ["Team A", "Team B", "Team A"],
        "opponent": ["Team B", "Team A", "Team B"],
        "venue": ["Home", "Away", "Home"],
        "round": ["Matchweek 1", "Matchweek 2", "Invalid"],
        "gf_for": [2, 0, 1],
        "ga_for": [1, 2, None],
        "xg_for": [1.5, 0.5, 0.8],
        "gf_against": [0, 3, 1],
        "ga_against": [1, 0, None],
        "xg_against": [0.9, 1.1, 0.7],
    }
    return pd.DataFrame(data)

# --- Tests for preprocess_data ---
import numbers

def test_preprocess_data_filters_and_renames(sample_raw_df):
    df_processed = preprocess_data(sample_raw_df, league_name="TEST")
    # Only one home match with valid date/team/opponent
    assert len(df_processed) == 1
    row = df_processed.iloc[0]
    # Renamed metadata
    assert "home_team" in df_processed.columns
    assert row["home_team"] == "Team A"
    assert row["away_team"] == "Team B"
    # Renamed numeric stats
    assert row["gf_home"] == 2
    assert row["xg_away"] == pytest.approx(0.9)
    # Round parsed to integer (including numpy integer)
    assert isinstance(row["round"], numbers.Integral)
    assert int(row["round"]) == 1
    # Result calculation
    assert row["result_home"] == 1

# Drop duplicates test follows

def test_preprocess_data_drop_duplicates():
    # Two identical matches → should collapse to one
    df = pd.DataFrame({
        "date": ["2025-07-29", "2025-07-29"],
        "time": ["18:00", "18:00"],
        "comp": ["TEST", "TEST"],
        "season": ["2025-2026", "2025-2026"],
        "team": ["TeamA", "TeamA"],
        "opponent": ["TeamB", "TeamB"],
        "venue": ["Home", "Home"],
        "round": ["Matchweek 1", "Matchweek 1"],
        "gf_for": [1, 1],
        "ga_for": [0, 0],
        "xg_for": [1.0, 1.0],
        "gf_against": [0, 0],
        "ga_against": [1, 1],
        "xg_against": [0.5, 0.5],
    })
    df_processed = preprocess_data(df, league_name="TEST")
    assert len(df_processed) == 1
    # Two identical matches → should collapse to one
    df = pd.DataFrame({
        "date": ["2025-07-29", "2025-07-29"],
        "time": ["18:00", "18:00"],
        "comp": ["TEST", "TEST"],
        "season": ["2025-2026", "2025-2026"],
        "team": ["TeamA", "TeamA"],
        "opponent": ["TeamB", "TeamB"],
        "venue": ["Home", "Home"],
        "round": ["Matchweek 1", "Matchweek 1"],
        "gf_for": [1, 1],
        "ga_for": [0, 0],
        "xg_for": [1.0, 1.0],
        "gf_against": [0, 0],
        "ga_against": [1, 1],
        "xg_against": [0.5, 0.5],
    })
    df_processed = preprocess_data(df, league_name="TEST")
    assert len(df_processed) == 1


def test_preprocess_data_dropna_date():
    # One missing date → dropped
    df = pd.DataFrame({
        "date": [None, "2025-07-29"],
        "time": ["18:00", "18:00"],
        "comp": ["TEST", "TEST"],
        "season": ["2025-2026", "2025-2026"],
        "team": ["TeamA", "TeamA"],
        "opponent": ["TeamB", "TeamB"],
        "venue": ["Home", "Home"],
        "round": ["Matchweek 1", "Matchweek 1"],
        "gf_for": [1, 1],
        "ga_for": [0, 0],
        "xg_for": [1.0, 1.0],
        "gf_against": [0, 0],
        "ga_against": [1, 1],
        "xg_against": [0.5, 0.5],
    })
    df_processed = preprocess_data(df, league_name="TEST")
    assert len(df_processed) == 1

# --- Tests for ensure_numeric ---

def test_ensure_numeric_coerces_and_preserves():
    df = pd.DataFrame({
        "a": ["1", "2", "not_a_number"],
        "b": ["3.5", "4.2", "5.1"],
    })
    df2 = ensure_numeric(df.copy(), cols=["a", "b"])
    # 'a': two valid and one NaN
    assert pd.api.types.is_numeric_dtype(df2["a"])
    assert pd.isna(df2["a"].iloc[2])
    # 'b': all floats
    assert df2["b"].iloc[0] == pytest.approx(3.5)
    assert pd.api.types.is_float_dtype(df2["b"])

# --- Tests for process_matches ---

def test_process_matches_writes_file_and_returns_df(tmp_path, monkeypatch, sample_raw_df):
    # Change working dir
    monkeypatch.chdir(tmp_path)
    import src.data.process as process_mod
    # Stub feature engineering
    def fake_add_all_features(df, stat_windows, agg_window):
        df_copy = df.copy()
        df_copy['dummy_feat'] = 42
        return df_copy
    monkeypatch.setattr(process_mod, 'add_all_features', fake_add_all_features)
    stat_windows = {"xg": [5], "gf": [5], "ga": [5]}
    df_out = process_matches(sample_raw_df, stat_windows, league_name="TEST")
    # Check stub column
    assert 'dummy_feat' in df_out.columns
    assert all(df_out['dummy_feat'] == 42)
    # Check file output
    file_path = tmp_path / 'data' / 'processed' / 'test_processed.csv'
    assert file_path.exists()
    df_saved = pd.read_csv(file_path)
    assert 'home_team' in df_saved.columns
    assert df_saved['dummy_feat'].iloc[0] == 42
