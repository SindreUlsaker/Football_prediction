import pandas as pd


def calculate_team_form_features(
    df: pd.DataFrame, stats: list[str], windows: list[int]
) -> pd.DataFrame:
    """
    Calculate rolling form features for given stats per team, independent of venue.
    Stats: list of base stat names, e.g. ["xg", "gf", "ga"]
    Windows: list of window sizes for rolling average, e.g. [5,10]
    Adds columns: "{stat}_home_roll{w}" and "{stat}_away_roll{w}"
    """
    df = df.copy()
    # Create long format: one row per team per match
    home = df[["date", "home_team"] + [f"{stat}_home" for stat in stats]].copy()
    home = home.rename(
        columns={"home_team": "team", **{f"{stat}_home": stat for stat in stats}}
    )
    away = df[["date", "away_team"] + [f"{stat}_away" for stat in stats]].copy()
    away = away.rename(
        columns={"away_team": "team", **{f"{stat}_away": stat for stat in stats}}
    )
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values("date")

    # Compute rolling for each stat and window
    for stat in stats:
        for w in windows:
            col = f"{stat}_roll{w}"
            long[col] = long.groupby("team")[stat].transform(
                lambda x: x.shift().rolling(w, min_periods=1).mean().round(2)
            )

    # Merge back for each window and stat for home/away teams
    for stat in stats:
        for w in windows:
            roll_col = f"{stat}_roll{w}"
            home_col = f"{stat}_home_roll{w}"
            away_col = f"{stat}_away_roll{w}"
            df = df.merge(
                long[["date", "team", roll_col]].rename(
                    columns={"team": "home_team", roll_col: home_col}
                ),
                on=["date", "home_team"],
                how="left",
            )
            df = df.merge(
                long[["date", "team", roll_col]].rename(
                    columns={"team": "away_team", roll_col: away_col}
                ),
                on=["date", "away_team"],
                how="left",
            )
    return df


def calculate_conceded_form_features(
    df: pd.DataFrame, windows: list[int]
) -> pd.DataFrame:
    """
    Calculate rolling form features for conceded stats per team, independent of venue.
    Adds columns: "xg_conceded_home_roll{w}" and "xg_conceded_away_roll{w}"
    """
    df = df.copy()
    # long format for conceded: home_team concedes xg_away, away_team concedes xg_home
    home = df[["date", "home_team", "xg_away"]].rename(
        columns={"home_team": "team", "xg_away": "xg_conceded"}
    )
    away = df[["date", "away_team", "xg_home"]].rename(
        columns={"away_team": "team", "xg_home": "xg_conceded"}
    )
    long = pd.concat([home, away], ignore_index=True).sort_values("date")

    # Rolling for conceded
    for w in windows:
        roll_col = f"roll{w}"
        long[roll_col] = long.groupby("team")["xg_conceded"].transform(
            lambda x: x.shift().rolling(w, min_periods=1).mean().round(2)
        )
        # Merge back
        home_col = f"xg_conceded_home_roll{w}"
        away_col = f"xg_conceded_away_roll{w}"
        df = df.merge(
            long[["date", "team", roll_col]].rename(
                columns={"team": "home_team", roll_col: home_col}
            ),
            on=["date", "home_team"],
            how="left",
        )
        df = df.merge(
            long[["date", "team", roll_col]].rename(
                columns={"team": "away_team", roll_col: away_col}
            ),
            on=["date", "away_team"],
            how="left",
        )
    return df


def calculate_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate static season-aggregated features per team, independent of venue.
    Adds columns for home and away teams: avg_points_home, avg_goals_for_home, avg_goals_against_home,
    avg_points_away, avg_goals_for_away, avg_goals_against_away, and home_advantage.
    """
    df = df.copy()
    # Points per match
    df["points_home"] = df["result_home"].map({1: 3, 0: 1, -1: 0})
    df["points_away"] = df["result_home"].map({1: 0, 0: 1, -1: 3})

    # Prepare long format for static
    home = df[["season", "home_team", "points_home", "gf_home", "ga_home"]].rename(
        columns={
            "home_team": "team",
            "points_home": "points",
            "gf_home": "goals_for",
            "ga_home": "goals_against",
        }
    )
    away = df[["season", "away_team", "points_away", "gf_away", "ga_away"]].rename(
        columns={
            "away_team": "team",
            "points_away": "points",
            "gf_away": "goals_for",
            "ga_away": "goals_against",
        }
    )
    long = pd.concat([home, away], ignore_index=True)

    # Aggregate per season and team
    agg = (
        long.groupby(["season", "team"])
        .agg(
            avg_points=("points", "mean"),
            avg_goals_for=("goals_for", "mean"),
            avg_goals_against=("goals_against", "mean"),
        )
        .round(2)
        .reset_index()
    )

    # Merge back for home and away teams
    df = df.merge(
        agg.rename(
            columns={
                "team": "home_team",
                "avg_points": "avg_points_home",
                "avg_goals_for": "avg_goals_for_home",
                "avg_goals_against": "avg_goals_against_home",
            }
        ),
        on=["season", "home_team"],
        how="left",
    )
    df = df.merge(
        agg.rename(
            columns={
                "team": "away_team",
                "avg_points": "avg_points_away",
                "avg_goals_for": "avg_goals_for_away",
                "avg_goals_against": "avg_goals_against_away",
            }
        ),
        on=["season", "away_team"],
        how="left",
    )

    # Home advantage remains unchanged
    df["home_advantage"] = (df["avg_points_home"] - df["avg_points_away"]).round(2)
    return df


def add_all_features(
    df: pd.DataFrame, stat_windows: dict[str, list[int]]
) -> pd.DataFrame:
    """
    Wrapper that runs all feature calculations:
      - Rolling form for stats (e.g. xg, gf, ga)
      - Rolling conceded form for xG against
      - Static season aggregates

    Parametre:
    - stat_windows: dict mapping stat names to list of window sizes,
      e.g. {'xg': [5,10], 'gf': [5], 'ga': [5]}
    """
    df = df.copy()
    # Rolling form
    stats = list(stat_windows.keys())
    windows = sorted({w for ws in stat_windows.values() for w in ws})
    df = calculate_team_form_features(df, stats, windows)
    # Conceded form (xG against)
    df = calculate_conceded_form_features(df, windows)
    # Static aggregations
    df = calculate_static_features(df)

    df["avg_points_diff"] = (df["avg_points_home"] - df["avg_points_away"]).round(2)
    df["avg_goals_for_diff"] = (
        df["avg_goals_for_home"] - df["avg_goals_for_away"]
    ).round(2)
    df["avg_goals_against_diff"] = (
        df["avg_goals_against_home"] - df["avg_goals_against_away"]
    ).round(2)

    # Rolling-diff for hver stat og hvert vindu
    for stat, windows in stat_windows.items():
        for w in windows:
            home_col = f"{stat}_home_roll{w}"
            away_col = f"{stat}_away_roll{w}"
            diff_col = f"{stat}_diff_roll{w}"
            df[diff_col] = (df[home_col] - df[away_col]).round(2)

    return df
