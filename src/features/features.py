import pandas as pd


def _compute_relegated_averages(
    df: pd.DataFrame, agg_prev: pd.DataFrame, spots: int = 3
) -> dict[str, dict[str, float]]:
    """
    Returnerer for hver sesong strengen 'YYYY-YYYY' et dict med
    gj.snitt mål-for og mål-imot for de nederste `spots` lagene.
    """
    # a) Poeng per kamp
    home = df[["season", "home_team", "result_home"]].rename(
        columns={"home_team": "team", "result_home": "res"}
    )
    home["points"] = home["res"].map({1: 3, 0: 1, -1: 0})

    away = df[["season", "away_team", "result_home"]].rename(
        columns={"away_team": "team", "result_home": "res"}
    )
    # For bortelag, invert resultat
    away["points"] = away["res"].map({1: 0, 0: 1, -1: 3})

    pts = pd.concat(
        [home[["season", "team", "points"]], away[["season", "team", "points"]]]
    )
    standings = pts.groupby(["season", "team"]).sum().reset_index()

    # b) Beregn gj.snitt GOALS fra agg_prev
    relegated_stats = {}
    for season, grp in standings.groupby("season"):
        bottom = grp.nsmallest(spots, "points")["team"].tolist()
        sub = agg_prev[(agg_prev["season"] == season) & (agg_prev["team"].isin(bottom))]
        relegated_stats[season] = {
            "for": sub["avg_goals_for_prev"].mean().round(2),
            "against": sub["avg_goals_against_prev"].mean().round(2),
        }
    return relegated_stats


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


from typing import Dict
import pandas as pd

# For beregning av nedrykksstats


def _compute_relegated_averages(
    df: pd.DataFrame, agg_prev: pd.DataFrame, spots: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Returnerer for hver sesong strengen 'YYYY-YYYY' et dict med
    gj.snitt mål-for og mål-imot for de nederste `spots` lagene.
    """
    # a) Poeng per kamp, hjemmelag
    home = df[["season", "home_team", "result_home"]].rename(
        columns={"home_team": "team", "result_home": "res"}
    )
    home["points"] = home["res"].map({1: 3, 0: 1, -1: 0})

    # b) Bortelag
    away = df[["season", "away_team", "result_home"]].rename(
        columns={"away_team": "team", "result_home": "res"}
    )
    away["points"] = away["res"].map({1: 0, 0: 1, -1: 3})

    # c) Sammenslå og summer poeng per sesong/team
    pts = pd.concat(
        [home[["season", "team", "points"]], away[["season", "team", "points"]]]
    )
    standings = pts.groupby(["season", "team"]).sum().reset_index()

    # d) Beregn gjennomsnittsmål for nedrykk
    relegated_stats: Dict[str, Dict[str, float]] = {}
    for season, grp in standings.groupby("season"):
        bottom_teams = grp.nsmallest(spots, "points")["team"].tolist()
        subset = agg_prev[
            (agg_prev["season"] == season) & (agg_prev["team"].isin(bottom_teams))
        ]
        relegated_stats[season] = {
            "for": round(subset["avg_goals_for_prev"].mean(), 2),
            "against": round(subset["avg_goals_against_prev"].mean(), 2),
        }
    return relegated_stats


def calculate_static_features(df: pd.DataFrame, agg_window: int) -> pd.DataFrame:
    """
    Beregner statiske features per kamp:
      - Fjorårsmål (hjemme/borte), med fyll for promoberte lag
      - Inneværende sesongs snittmål
      - Matches_played per lag
      - Vektet miks av fjorår og inneværende sesong
      - Home_advantage

    For nyopprykkede lag fylles fjorårsmål med snittverdier fra lagene
    som rykket ned i forrige sesong, _dersom_ statistikk for den sesongen finnes.
    """
    df = df.copy()

    # 1) Standard fjorårsgjennomsnitt per lag
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

    # 2) Beregn gj.snitt for nedrykkslag per sesong
    relegated_stats = _compute_relegated_averages(df, agg_prev, spots=3)

    # 3) Hjelpekolonne: prev_season (forrige sesong-streng)
    df["prev_season"] = (
        df["season"]
        .astype(str)
        .apply(lambda s: f"{int(s.split('-')[0]) - 1}-{int(s.split('-')[1]) - 1}")
    )

    # 4) Merge fjorårsmål for hjemmelag og bortelag
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

    # Fjern dupliserte season-kolonner, behold kun venstre
    df = (
        df.drop(columns=["season"])
        .rename(columns={"season_x": "season"})
        .drop(columns=["season_y"])
    )

    # 5) Fyll på for nyopprykkede _kun_ om prev_season finnes i relegated_stats
    valid_prev = df["prev_season"].isin(relegated_stats)

    mask_home = df["avg_goals_for_prev_home"].isna() & valid_prev
    df.loc[mask_home, "avg_goals_for_prev_home"] = df.loc[mask_home, "prev_season"].map(
        lambda s: relegated_stats[s]["for"]
    )
    df.loc[mask_home, "avg_goals_against_prev_home"] = df.loc[
        mask_home, "prev_season"
    ].map(lambda s: relegated_stats[s]["against"])

    mask_away = df["avg_goals_for_prev_away"].isna() & valid_prev
    df.loc[mask_away, "avg_goals_for_prev_away"] = df.loc[mask_away, "prev_season"].map(
        lambda s: relegated_stats[s]["for"]
    )
    df.loc[mask_away, "avg_goals_against_prev_away"] = df.loc[
        mask_away, "prev_season"
    ].map(lambda s: relegated_stats[s]["against"])

    # 6) Fjern hjelpekolonnen
    df = df.drop(columns=["prev_season"])

    # 7) Nåværende sesong stats (uendret)
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

    stats_curr = (
        long_curr.groupby(["season", "team"])
        .agg(
            avg_goals_for_curr=("goals_for", "mean"),
            avg_goals_against_curr=("goals_against", "mean"),
        )
        .round(2)
        .reset_index()
    )

    df = df.merge(
        stats_curr.rename(
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
        stats_curr.rename(
            columns={
                "team": "away_team",
                "avg_goals_for_curr": "avg_goals_for_curr_away",
                "avg_goals_against_curr": "avg_goals_against_curr_away",
            }
        ),
        on=["season", "away_team"],
        how="left",
    )

    # 8) matches_played
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

    df = df.merge(
        hist.rename(
            columns={"team": "home_team", "matches_played": "matches_played_home"}
        ),
        on=["season", "home_team", "date"],
        how="left",
    )
    df = df.merge(
        hist.rename(
            columns={"team": "away_team", "matches_played": "matches_played_away"}
        ),
        on=["season", "away_team", "date"],
        how="left",
    )

    # 9) Vekting av fjorår vs curr
    def weighted(prev, curr, played):
        if played == 0 and pd.notna(prev):
            return round(prev, 2)
        if pd.isna(prev):
            return round(curr, 2)
        w = min(played / agg_window, 1)
        return round(w * curr + (1 - w) * prev, 2)

    for prefix in ["home", "away"]:
        df[f"avg_goals_for_{prefix}"] = df.apply(
            lambda r: weighted(
                r[f"avg_goals_for_prev_{prefix}"],
                r[f"avg_goals_for_curr_{prefix}"],
                r[f"matches_played_{prefix}"],
            ),
            axis=1,
        )
        df[f"avg_goals_against_{prefix}"] = df.apply(
            lambda r: weighted(
                r[f"avg_goals_against_prev_{prefix}"],
                r[f"avg_goals_against_curr_{prefix}"],
                r[f"matches_played_{prefix}"],
            ),
            axis=1,
        )

    # 10) Home-advantage
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
