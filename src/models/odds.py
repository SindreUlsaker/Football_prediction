# File: src/models/odds.py
import pandas as pd
import numpy as np
from scipy.stats import poisson
from src.models.predict import (
    load_models_for_league,
    compute_match_outcome_probabilities,
)  # :contentReference[oaicite:0]{index=0}


def _get_lambdas(
    df: pd.DataFrame,
    features_home: list[str],
    features_away: list[str],
    league: str,
    models_dir: str,
) -> tuple[float, float]:
    """
    Laster modell + scaler og returnerer (lam_h, lam_a) for én kamp,
    ved å align’e X_all etter scaler.feature_names_in_.
    """

    # 1) Last inn modell og scaler
    model, scaler = load_models_for_league(league, models_dir)

    # 2) Hent hjemmelags-features og bytt navn
    Xh = df[features_home].copy()
    # Fjern _home og _away på alle kolonner – slik vi gjorde i train_poisson_model :contentReference[oaicite:1]{index=1}
    Xh.columns = [c.replace("_home", "").replace("_away", "") for c in Xh.columns]
    Xh["is_home"] = 1

    # 3) Hent bortelags-features og bytt navn
    Xa = df[features_away].copy()
    Xa.columns = [c.replace("_away", "").replace("_home", "") for c in Xa.columns]
    Xa["is_home"] = 0

    # 4) Slå sammen og fyll missing
    X_all = pd.concat([Xh, Xa], ignore_index=True).fillna(0)

    # 5) Ta bare de kolonnene scaler faktisk forventer, i riktig rekkefølge
    X_all = X_all.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # 6) Skaler og prediker
    X_scaled = scaler.transform(X_all)
    lambdas = model.predict(X_scaled)

    return lambdas[0], lambdas[1]


def calculate_hub_odds(
    df: pd.DataFrame,
    features_home: list[str],
    features_away: list[str],
    league: str,
    models_dir: str,
    max_goals: int = 10,
) -> pd.DataFrame:
    """
    Returnerer DataFrame med sannsynlighet og fair odds for Hjemme/Uavgjort/Borte.
    """
    lam_h, lam_a = _get_lambdas(df, features_home, features_away, league, models_dir)
    p_h, p_d, p_a = compute_match_outcome_probabilities(
        lam_h, lam_a, max_goals
    )  # :contentReference[oaicite:2]{index=2}
    odds = {
        "Hjemmeseier": (p_h, 1 / p_h),
        "Uavgjort": (p_d, 1 / p_d),
        "Borteseier": (p_a, 1 / p_a),
    }
    return pd.DataFrame(
        [
            {
                "Utfall": k,
                "Sannsynlighet": f"{v[0]*100:.1f}%",
                "Fair odds": f"{v[1]:.2f}",
            }
            for k, v in odds.items()
        ]
    )


def calculate_btts_odds(
    df: pd.DataFrame,
    features_home: list[str],
    features_away: list[str],
    league: str,
    models_dir: str,
) -> pd.DataFrame:
    """
    Returnerer sannsynlighet og fair odds for “Begge lag scorer – Ja/Nei”.
    """
    lam_h, lam_a = _get_lambdas(df, features_home, features_away, league, models_dir)
    p0_h = poisson.pmf(0, lam_h)
    p0_a = poisson.pmf(0, lam_a)
    p_yes = (1 - p0_h) * (1 - p0_a)
    p_no = 1 - p_yes
    return pd.DataFrame(
        [
            {
                "Utfall": "Begge lag scorer – Ja",
                "Sannsynlighet": f"{p_yes*100:.1f}%",
                "Fair odds": f"{1/p_yes:.2f}",
            },
            {
                "Utfall": "Begge lag scorer – Nei",
                "Sannsynlighet": f"{p_no*100:.1f}%",
                "Fair odds": f"{1/p_no:.2f}",
            },
        ]
    )


def calculate_over_under_odds(
    df: pd.DataFrame,
    features_home: list[str],
    features_away: list[str],
    league: str,
    models_dir: str,
    threshold: int = 2.5,
) -> pd.DataFrame:
    """
    Returnerer sannsynlighet og fair odds for Over/Under gitt antall mål (f.eks. 2.5).
    """
    lam_h, lam_a = _get_lambdas(df, features_home, features_away, league, models_dir)
    # Sannsynlighet for totalt mål ≤ threshold = sum_{i+j ≤ T} P(i,j)
    max_g = int(np.ceil(threshold)) + 5
    prob_matrix = np.array(
        [
            [poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a) for j in range(max_g + 1)]
            for i in range(max_g + 1)
        ]
    )
    p_under = prob_matrix[
        np.add.outer(np.arange(max_g + 1), np.arange(max_g + 1)) <= threshold
    ].sum()
    p_over = 1 - p_under
    return pd.DataFrame(
        [
            {
                "Utfall": f"Under {threshold}",
                "Sannsynlighet": f"{p_under*100:.1f}%",
                "Fair odds": f"{1/p_under:.2f}",
            },
            {
                "Utfall": f"Over {threshold}",
                "Sannsynlighet": f"{p_over*100:.1f}%",
                "Fair odds": f"{1/p_over:.2f}",
            },
        ]
    )
