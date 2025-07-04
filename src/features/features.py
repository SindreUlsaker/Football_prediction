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


def calculate_static_features(df: pd.DataFrame, agg_window: int) -> pd.DataFrame:
    """
    Calculate static season-aggregated features per team, independent of venue.
    Adds columns for home and away teams:
      avg_goals_for_home, avg_goals_against_home,
      avg_goals_for_away, avg_goals_against_away,
    and home_advantage (based on goals).
    """
    df = df.copy()

    # 1) Fjorårsgjennomsnitt per lag ved å justere sesong-strengen
    prev = df.copy()

    home_prev = prev.rename(
        columns={
            "home_team": "team",
            "gf_home": "goals_for",
            "ga_home": "goals_against",
        }
    )[["season", "team", "goals_for", "goals_against"]]
    away_prev = prev.rename(
        columns={
            "away_team": "team",
            "gf_away": "goals_for",
            "ga_away": "goals_against",
        }
    )[["season", "team", "goals_for", "goals_against"]]

    agg_prev = (
        pd.concat([home_prev, away_prev])
        .groupby(["season", "team"])
        .mean()
        .round(2)
        .reset_index()
        .rename(
            columns={
                "goals_for": "avg_goals_for_prev",
                "goals_against": "avg_goals_against_prev",
            }
        )
    )

    # Hjelpe-kolonne: hvilken sesong henter vi prev‐stats fra?
    df["prev_season"] = (
        df["season"]
        .astype(str)
        .apply(lambda s: f"{int(s.split('-')[0]) - 1}-{int(s.split('-')[1]) - 1}")
    )

    # Merge
    df = df.merge(
        agg_prev.rename(
            columns={
                "team": "home_team",
                "avg_goals_for_prev": "avg_goals_for_prev_home",
                "avg_goals_against_prev": "avg_goals_against_prev_home",
            }
        ),
        left_on=["prev_season", "home_team"],
        right_on=["season", "home_team"],
        how="left",
    )
    df = df.merge(
        agg_prev.rename(
            columns={
                "team": "away_team",
                "avg_goals_for_prev": "avg_goals_for_prev_away",
                "avg_goals_against_prev": "avg_goals_against_prev_away",
            }
        ),
        left_on=["prev_season", "away_team"],
        right_on=["season", "away_team"],
        how="left",
    )

    # Rydd opp i season-kolonnene: behold bare season_x (venstre nøkkel), og kalle den season
    df = df.drop(columns=["season"])  # fjerner den opprinnelige season
    df = df.rename(columns={"season_x": "season"})  # gir season_x tilbake navnet season
    df = df.drop(columns=["season_y", "prev_season"])  # fjerner den høyre nøkkelen og hjelpe-kolonnen

    # 3) Statistikk for nåværende sesong – samme gjennomsnitt på alle kamper

    # a) Bygg long‐format
    home_curr = df[["season", "home_team", "gf_home", "ga_home"]].rename(
        columns={
            "home_team": "team",
            "gf_home": "goals_for",
            "ga_home": "goals_against",
        }
    )
    away_curr = df[["season", "away_team", "gf_away", "ga_away"]].rename(
        columns={
            "away_team": "team",
            "gf_away": "goals_for",
            "ga_away": "goals_against",
        }
    )
    long_curr = pd.concat([home_curr, away_curr], ignore_index=True)

    # b) Aggreger per-lag per-sesong
    stats = (
        long_curr.groupby(["season", "team"])
        .agg(
            avg_goals_for_curr=("goals_for", "mean"),
            avg_goals_against_curr=("goals_against", "mean"),
        )
        .reset_index()
    )
    stats["avg_goals_for_curr"] = stats["avg_goals_for_curr"].round(2)
    stats["avg_goals_against_curr"] = stats["avg_goals_against_curr"].round(2)

    # c) Merge snapshot stats for home and away teams
    df = df.merge(
        stats.rename(
            columns={
                "team": "home_team",
                "avg_goals_for_curr": "avg_goals_for_curr_home",
                "avg_goals_against_curr": "avg_goals_against_curr_home",
            }
        ),
        on=["season", "home_team"],
        how="left",
    )
    df = df.merge(
        stats.rename(
            columns={
                "team": "away_team",
                "avg_goals_for_curr": "avg_goals_for_curr_away",
                "avg_goals_against_curr": "avg_goals_against_curr_away",
            }
        ),
        on=["season", "away_team"],
        how="left",
    )

    # --- Dynamisk matches_played per kamp (teller kun kamper spilt før denne kampen) ---
    home_hist = df[["date", "season", "home_team"]].rename(
        columns={"home_team": "team"}
    )
    away_hist = df[["date", "season", "away_team"]].rename(
        columns={"away_team": "team"}
    )
    hist = pd.concat([home_hist, away_hist], ignore_index=True).sort_values(
        ["team", "season", "date"]
    )
    hist["matches_played"] = hist.groupby(["team", "season"]).cumcount()

    # Merge dynamisk antall kamper spilt før denne kampen
    df = df.merge(
        hist.rename(
            columns={
                "team": "home_team",
                "matches_played": "matches_played_home",
            }
        ),
        on=["season", "home_team", "date"],
        how="left",
    )
    df = df.merge(
        hist.rename(
            columns={
                "team": "away_team",
                "matches_played": "matches_played_away",
            }
        ),
        on=["season", "away_team", "date"],
        how="left",
    )

    # 4) Vekta kombinasjon av fjorår og inneværende sesong
    def weighted(prev, curr, played):
        # 1) Hvis ingen kamper spilt, og vi har fjorårstall: bruk fjorårstall (rundet)
        if played == 0 and pd.notna(prev):
            return round(prev, 2)
        # 2) Hvis ingen fjorårstall (nyopprykket): bruk bare inneværende sesong
        if pd.isna(prev):
            return round(curr, 2)
        # 3) Ellers gjør vanlig lineær vektet miks
        w = min(played / agg_window, 1)
        return round(w * curr + (1 - w) * prev, 2)

    df["avg_goals_for_home"] = df.apply(
        lambda r: weighted(
            r["avg_goals_for_prev_home"],
            r["avg_goals_for_curr_home"],
            r["matches_played_home"],
        ),
        axis=1,
    )
    df["avg_goals_against_home"] = df.apply(
        lambda r: weighted(
            r["avg_goals_against_prev_home"],
            r["avg_goals_against_curr_home"],
            r["matches_played_home"],
        ),
        axis=1,
    )
    df["avg_goals_for_away"] = df.apply(
        lambda r: weighted(
            r["avg_goals_for_prev_away"],
            r["avg_goals_for_curr_away"],
            r["matches_played_away"],
        ),
        axis=1,
    )
    df["avg_goals_against_away"] = df.apply(
        lambda r: weighted(
            r["avg_goals_against_prev_away"],
            r["avg_goals_against_curr_away"],
            r["matches_played_away"],
        ),
        axis=1,
    )

    # 5) Home advantage basert på mål
    df["home_advantage"] = (df["avg_goals_for_home"] - df["avg_goals_for_away"]).round(
        2
    )

    return df


def add_all_features(
    df: pd.DataFrame, stat_windows: dict[str, list[int]], agg_window: int
) -> pd.DataFrame:
    """
    Wrapper that runs all feature calculations:
      - Rolling form for stats (e.g. xg, gf, ga)
      - Rolling conceded form for xG against
      - Static season aggregates

    Parametre:
    - stat_windows: dict mapping stat names to list of window sizes,
      e.g. {'xg': [5,10], 'gf': [5], 'ga': [5]}
    - agg_window: hvor mange kamper som brukes i vektet sesong‐gjennomsnitt
    """
    df = df.copy()

    # 1) Rolling form
    stats = list(stat_windows.keys())
    windows = sorted({w for ws in stat_windows.values() for w in ws})
    df = calculate_team_form_features(df, stats, windows)

    # 2) Conceded form (xG against)
    df = calculate_conceded_form_features(df, windows)

    # 3) Static season aggregates (inkl. goals‐for / goals‐against, vektet mot fjorår)
    df = calculate_static_features(df, agg_window=agg_window)

    return df
