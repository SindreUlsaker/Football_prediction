# File: src/models/simulate.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

from config.settings import DATA_PATH
from src.models.predict import predict_poisson_from_models


# Hold disse i sync med øvrige sider (predictions/oddschecker)
STAT_WINDOWS = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}


def _build_feature_lists() -> Tuple[list[str], list[str]]:
    """
    Bygger feature-lister konsistent med øvrig pipeline.
    Offensivt for laget som angriper, defensivt for motstander.
    """
    features_home = (
        [f"xg_home_roll{w}" for w in STAT_WINDOWS["xg"]]
        + [f"gf_home_roll{w}" for w in STAT_WINDOWS["gf"]]
        + [f"xg_conceded_away_roll{w}" for w in STAT_WINDOWS["xg"]]
        + [f"ga_away_roll{w}" for w in STAT_WINDOWS["ga"]]
        + ["avg_goals_for_home", "avg_goals_against_away"]
    )
    features_away = (
        [f"xg_away_roll{w}" for w in STAT_WINDOWS["xg"]]
        + [f"gf_away_roll{w}" for w in STAT_WINDOWS["gf"]]
        + [f"xg_conceded_home_roll{w}" for w in STAT_WINDOWS["xg"]]
        + [f"ga_home_roll{w}" for w in STAT_WINDOWS["ga"]]
        + ["avg_goals_for_away", "avg_goals_against_home"]
    )
    return features_home, features_away


def _latest_season_str(seasons: pd.Series) -> str:
    """
    Finn siste sesong gitt strenger 'YYYY-YYYY' ved å sortere på første årstall.
    """

    def key(s: str) -> int:
        try:
            return int(str(s).split("-")[0])
        except Exception:
            return -1

    uniq = sorted(seasons.dropna().unique(), key=key)
    if not uniq:
        raise ValueError("Fant ingen sesongverdier i processed-data.")
    return uniq[-1]


def _current_points(played: pd.DataFrame) -> pd.Series:
    """
    Returnerer poengsum per lag basert på spilte kamper i `played`.
    Bruker 'result_home' (1/0/-1) hvis tilgjengelig; faller tilbake til gf/ga.
    """
    df = played.copy()

    has_result = "result_home" in df.columns and df["result_home"].notna().any()
    if not has_result:
        # Fallback: avled resultater fra score
        needed = {"gf_home", "ga_home"}
        if not needed.issubset(df.columns):
            raise ValueError(
                "Mangler både 'result_home' og (gf_home, ga_home) i processed-data."
            )
        df["result_home"] = (df["gf_home"] > df["ga_home"]).astype(int)
        df.loc[df["gf_home"] == df["ga_home"], "result_home"] = 0
        df.loc[df["gf_home"] < df["ga_home"], "result_home"] = -1

    # Poeng fra hjemmeperspektiv
    home_pts_map = {1: 3, 0: 1, -1: 0}
    away_pts_map = {1: 0, 0: 1, -1: 3}

    hp = df[["home_team", "result_home"]].rename(
        columns={"home_team": "team", "result_home": "res"}
    )
    hp["points"] = hp["res"].map(home_pts_map)

    ap = df[["away_team", "result_home"]].rename(
        columns={"away_team": "team", "result_home": "res"}
    )
    ap["points"] = ap["res"].map(away_pts_map)

    pts = pd.concat([hp[["team", "points"]], ap[["team", "points"]]])
    all_teams = pd.Index(sorted(set(df["home_team"]).union(df["away_team"])))

    # Summer per lag og fyll 0 for lag uten spilte kamper
    return (
        pts.groupby("team")["points"]
        .sum()
        .reindex(all_teams)
        .fillna(0)
        .astype(int)
        .sort_index()
    )


def _simulate_once(
    preds: pd.DataFrame,
    base_points: pd.Series,
    rng: np.random.Generator,
    relegation_spots: int,
    top_n: int,
) -> tuple[set[str], set[str], set[str]]:
    """
    Én simulering: trekker utfall for hver gjenstående kamp og rangerer lagene.
    `preds` har kolonner: home_team, away_team, prob_home, prob_draw, prob_away.
    Returnerer (champions, topN, relegated) som sett av lagnavn.
    """
    points = base_points.copy()

    for _, row in preds.iterrows():
        p = np.array(
            [row["prob_home"], row["prob_draw"], row["prob_away"]], dtype=float
        )
        if not np.all(np.isfinite(p)) or p.sum() <= 0:
            # Beskyttelse mot NaN/inf/negativ sum: fall tilbake til jevnt
            p = np.array([1.0, 1.0, 1.0])
        p = p / p.sum()

        outcome = rng.choice([0, 1, 2], p=p)  # 0=H, 1=U, 2=B
        if outcome == 0:
            points[row["home_team"]] = points.get(row["home_team"], 0) + 3
        elif outcome == 1:
            points[row["home_team"]] = points.get(row["home_team"], 0) + 1
            points[row["away_team"]] = points.get(row["away_team"], 0) + 1
        else:
            points[row["away_team"]] = points.get(row["away_team"], 0) + 3

    # Tie-break: jitter for deterministisk rangering ved poenglikhet
    teams = points.index.to_list()
    jitter = rng.uniform(0, 1e-6, size=len(teams))
    scores = points.values + jitter
    order = np.argsort(-scores)  # høyest først
    ranked = [teams[i] for i in order]

    champion = {ranked[0]}
    topN = set(ranked[:top_n])
    relegated = set(ranked[-relegation_spots:]) if relegation_spots > 0 else set()
    return champion, topN, relegated


def run_simulations(
    league_name: str,
    n_sims: int = 1000,
    season: str | None = None,
    top_n: int = 5,
    relegation_spots: int = 3,
    models_dir: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Kjør Monte Carlo-simulering for valgt liga/sesong.

    Returnerer DataFrame med kolonner:
      - Team
      - P(vinne)   (i %)
      - P(topp N)  (i %)
      - P(nedrykk) (i %)
    """
    # --- Les processed ---
    key = league_name.lower().replace(" ", "_")
    processed_path = f"{DATA_PATH}/processed/{key}_processed.csv"
    df = pd.read_csv(processed_path, parse_dates=["date"])

    # --- Velg sesong ---
    season = season or _latest_season_str(df["season"])
    df_season = df[df["season"] == season].copy()

    # --- Splitt spilte vs gjenstående ---
    played = df_season[df_season["result_home"].notna()].copy()
    remaining = df_season[df_season["result_home"].isna()].copy()

    # --- Startpoeng (låser historikk) ---
    base_points = _current_points(played)

    # --- Alle lag (for å sikre komplett output) ---
    all_teams = pd.Index(
        sorted(set(df_season["home_team"]).union(df_season["away_team"]))
    )

    # --- Håndter sesong ferdig ---
    if remaining.empty:
        out = pd.DataFrame({"Team": all_teams})
        out["P(vinne)"] = 0.0
        out[f"P(topp {top_n})"] = 0.0
        out["P(nedrykk)"] = 0.0
        if not base_points.empty:
            leader = base_points.reindex(all_teams).fillna(0).idxmax()
            out.loc[out["Team"] == leader, "P(vinne)"] = 100.0
        return out.sort_values("P(vinne)", ascending=False).reset_index(drop=True)

    # --- Bygg features og prediker sannsynligheter for gjenstående kamper (én gang) ---
    features_home, features_away = _build_feature_lists()
    if models_dir is None:
        models_dir = f"{DATA_PATH}/models"

    preds = predict_poisson_from_models(
        df=remaining,
        features_home=features_home,
        features_away=features_away,
        league_name=league_name,
        models_dir=models_dir,
        max_goals=10,
        boost=False,
    )[["home_team", "away_team", "prob_home", "prob_draw", "prob_away"]]

    # --- Init aggregering ---
    teams = sorted(set(all_teams))
    champ_count = {t: 0 for t in teams}
    topN_count = {t: 0 for t in teams}
    releg_count = {t: 0 for t in teams}

    rng = np.random.default_rng(seed)

    # --- Monte Carlo ---
    for _ in range(int(n_sims)):
        champion, topN, relegated = _simulate_once(
            preds=preds,
            base_points=base_points,
            rng=rng,
            relegation_spots=int(relegation_spots),
            top_n=int(top_n),
        )
        for t in champion:
            champ_count[t] += 1
        for t in topN:
            topN_count[t] += 1
        for t in relegated:
            releg_count[t] += 1

    # --- Til tabell (prosent) ---
    out = pd.DataFrame({"Team": teams})
    out["P(vinne)"] = [round(100.0 * champ_count[t] / n_sims, 1) for t in teams]
    out[f"P(topp {top_n})"] = [round(100.0 * topN_count[t] / n_sims, 1) for t in teams]
    out["P(nedrykk)"] = [round(100.0 * releg_count[t] / n_sims, 1) for t in teams]

    return out.sort_values("P(vinne)", ascending=False).reset_index(drop=True)
