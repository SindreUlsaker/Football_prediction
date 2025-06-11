import pandas as pd


def predict_future_matches(
    model, scaler, future_matches: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """
    Gjør prediksjoner for kommende kamper og beregner fair odds.

    Parametre:
    - model: trent modell med .predict og .predict_proba
    - scaler: objekt med .transform (samme som dere brukte i trening)
    - future_matches: DataFrame med kolonner 'date', 'home_team', 'away_team' + alle feature-kolonner
    - features: liste av feature-kolonnenavn som skal brukes til X

    Returnerer:
    - DataFrame med kolonner:
        ['date','home_team','away_team',
         'prediction','prob_home','prob_draw','prob_away',
         'fair_odds_home','fair_odds_draw','fair_odds_away']
    """
    # 1) Lag X-matrise
    X = scaler.transform(future_matches[features])

    # 2) Prediksjoner og sannsynligheter
    preds = model.predict(X)
    probas = model.predict_proba(X)
    classes = list(model.classes_)

    # 3) Bygg resultat-DF
    result = future_matches[["date", "home_team", "away_team"]].copy()
    result["prediction"] = preds
    # utknekking av sannsynligheter
    result["prob_home"] = [round(p[classes.index(1)], 2) for p in probas]
    result["prob_draw"] = [round(p[classes.index(0)], 2) for p in probas]
    result["prob_away"] = [round(p[classes.index(-1)], 2) for p in probas]

    # 4) Fair odds = 1 / sannsynlighet
    for side in ["home", "draw", "away"]:
        result[f"fair_odds_{side}"] = result[f"prob_{side}"].apply(
            lambda p: round(1 / p, 2) if p and p > 0 else None
        )

    return result
