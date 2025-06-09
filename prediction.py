import pandas as pd

def predict_future_matches(model, future_matches, features_dict):
    predictions = []

    # Iterer over hver match i future_matches
    for _, match in future_matches.iterrows():
        home_team = match['home']
        away_team = match['away']

        # Hent features for både hjemme- og bortelag
        home_features = features_dict.get(home_team, {})
        away_features = features_dict.get(away_team, {})

        # Sjekk at vi har alle nødvendige features
        if not home_features or not away_features:
            print(f"Missing features for {home_team} or {away_team}. Skipping prediction.")
            continue

        # Samle features for modellen i riktig rekkefølge
        model_input = [
            home_features['standard_gls'], home_features['standard_sot'], home_features['standard_dist'], home_features['expected_xg'],
            home_features['xg_rolling_avg'], home_features['xg.1_rolling_avg'], home_features['result_rolling_avg'],
            away_features['standard_gls'], away_features['standard_sot'], away_features['standard_dist'], away_features['expected_xg'],
            away_features['xg_rolling_avg'], away_features['xg.1_rolling_avg'], away_features['result_rolling_avg']
        ]

        # Konverter model_input til en DataFrame med riktige kolonnenavn
        model_input_df = pd.DataFrame(
            [model_input],
            columns=[
                "standard_gls",
                "standard_sot",
                "standard_dist",
                "expected_xg",
                "xg_home_rolling_avg",
                "xg.1_home_rolling_avg",
                "result_home_rolling_avg",  # Endret navn her
                "standard_gls_away",
                "standard_sot_away",
                "standard_dist_away",
                "expected_xg_away",
                "xg_away_rolling_avg",
                "xg.1_away_rolling_avg",
                "result_away_rolling_avg",
            ],
        )

        # Gjør prediksjon med DataFrame
        prediction = model.predict(model_input_df)[0]

        # Mapper den numeriske prediksjonen til tekst
        if prediction == 1:
            result = 'Hjemmeseier'
        elif prediction == 0:
            result = 'Uavgjort'
        elif prediction == -1:
            result = 'Borteseier'
        else:
            result = 'Ukjent'  # Håndterer eventuelle uventede verdier

        predictions.append({
            'home_team': home_team,
            'away_team': away_team,
            'prediction': result
        })

    # Konverter resultatene til en DataFrame for oversikt
    predictions_df = pd.DataFrame(predictions)

    return predictions_df
