import pandas as pd
from data_processing import preprocess_data, calculate_rolling_averages, calculate_result_rolling_averages, calculate_match_results
from model_training import train_model
from prediction import predict_future_matches
from data_fetching import save_data_to_csv

def main():
    round_number = 38.0
    save_data_to_csv()
    
    # Les data fra CSV-filer
    matches = pd.read_csv("premier_league_matches.csv", usecols=["date", "home", "xg", "score", "xg.1", "away", "wk"])
    shooting_data = pd.read_csv("premier_league_shooting_stats.csv", usecols=["unnamed: 0_level_0_squad", "standard_gls", "standard_sot", "standard_dist", "expected_xg"])
    
    # Forbehandle data
    processed_data, shooting_data = preprocess_data(matches, shooting_data)

    # Beregn rolling averages
    rolling_stats = ['xg', 'xg.1']  # Kolonner som brukes til rolling averages
    processed_data = calculate_rolling_averages(processed_data, rolling_stats)
    processed_data = calculate_result_rolling_averages(processed_data)
    processed_data = calculate_match_results(processed_data)
    
    print(processed_data.head())  # Sjekk de første radene i dataen
    
    # Lag et dictionary som inneholder alle nødvendige features per lag
    features_dict = {}

    # Iterer over shooting_data
    for _, row in shooting_data.iterrows():
        team = row['team']  # Lagnavn
        
        team_data = processed_data[processed_data['home'] == team].sort_values('date', ascending=False)
    
        # Hent ut den siste tilgjengelige rolling average for hjemmelaget
        xg_rolling_avg = team_data['xg_home_rolling_avg'].iloc[0] if not team_data.empty else None
        xg_1_rolling_avg = team_data['xg.1_home_rolling_avg'].iloc[0] if not team_data.empty else None
        result_rolling_avg = team_data['result_home_rolling_avg'].iloc[0] if not team_data.empty else None 
        
        features_dict[team] = {
            'standard_gls': row['standard_gls'],
            'standard_sot': row['standard_sot'],
            'standard_dist': row['standard_dist'],
            'expected_xg': row['expected_xg'],
            'xg_rolling_avg': xg_rolling_avg,
            'xg.1_rolling_avg': xg_1_rolling_avg,
            'result_rolling_avg': result_rolling_avg
        }

    # Tren modell
    features = [ 
    'standard_gls', 'standard_sot', 'standard_dist', 'expected_xg',  
    'xg_home_rolling_avg', 'xg.1_home_rolling_avg', 'result_home_rolling_avg',
    'standard_gls_away', 'standard_sot_away', 'standard_dist_away', 'expected_xg_away',
    'xg_away_rolling_avg', 'xg.1_away_rolling_avg', 'result_away_rolling_avg'
    ]
    target = 'result_home'  # Kolonnen som representerer kampresultatet

    model = train_model(processed_data, features, target)

    print("Model trained successfully.")

    # Filtrer fremtidige kamper for ønsket runde
    future_matches = matches[
    (matches['wk'] == round_number)]
    if future_matches.empty:
        print(f"No matches found for round {round_number}.")
        return
    
    future_predictions = predict_future_matches(model, future_matches, features_dict)

    # Skriv ut prediksjoner
    print(f"Future match predictions:\n{future_predictions[['home_team', 'away_team', 'prediction']]}")

if __name__ == "__main__":
    main()
