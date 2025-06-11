import pandas as pd


def calculate_rolling_features(
    df: pd.DataFrame, stats: list[str], windows: list[int]
) -> pd.DataFrame:
    """
    Generiske rolling features for valgte statser og vinduer.
    """
    df = df.sort_values("date")
    for stat in stats:
        grp = "home_team" if stat.endswith("_home") else "away_team"
        for w in windows:
            col = f"{stat}_roll{w}"
            df[col] = (
                df.groupby(grp)[stat]
                .transform(lambda x: x.shift().rolling(w, min_periods=1).mean())
                .round(2)
            )
    return df


def calculate_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sesong-aggregerte features (poeng, mål for/ imot, hjemmebanefordel).
    """
    df = df.copy()
    # Poeng per kamp
    df["points_home"] = df["result_home"].map({1: 3, 0: 1, -1: 0})
    df["points_away"] = df["result_home"].map({1: 0, 0: 1, -1: 3})

    # Aggregater per sesong + lag
    home_agg = (
        df.groupby(["season", "home_team"])
        .agg(
            avg_points_home=("points_home", "mean"),
            avg_goals_for_home=("gf_home", "mean"),
            avg_goals_against_home=("ga_home", "mean"),
        )
        .reset_index()
    )
    away_agg = (
        df.groupby(["season", "away_team"])
        .agg(
            avg_points_away=("points_away", "mean"),
            avg_goals_for_away=("gf_away", "mean"),
            avg_goals_against_away=("ga_away", "mean"),
        )
        .reset_index()
    )

    # Merge tilbake
    df = df.merge(home_agg, on=["season", "home_team"], how="left")
    df = df.merge(away_agg, on=["season", "away_team"], how="left")

    # Hjemmebanefordel
    df["home_advantage"] = df["avg_points_home"] - df["avg_points_away"]
    return df


def add_all_features(
    df: pd.DataFrame, rolling_stats: list[str], rolling_windows: list[int]
) -> pd.DataFrame:
    """
    Wrapper som kjører både rolling- og statiske features.
    """
    df = calculate_rolling_features(df, rolling_stats, rolling_windows)
    df = calculate_static_features(df)
    return df
