import pandas as pd

def preprocess_data(matches, shooting_data):
    # Gi lagnavn-kolonnen i shooting_stats et mer beskrivende navn
    shooting_data.rename(columns={"unnamed: 0_level_0_squad": "team"}, inplace=True)
    
    # Kombiner hjemme- og bortelagsstatistikk med kampdata
    matches = matches.merge(
        shooting_data, left_on="home", right_on="team", suffixes=("", "_home")
    )
    matches = matches.merge(
        shooting_data, left_on="away", right_on="team", suffixes=("", "_away")
    )

    # Konverter datoen til datetime-format for å kunne sortere etter kampdato
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values(by="date")  # Sorter kampene etter dato

    matches.dropna(inplace=True)  # Fjern rader med manglende verdier
    return matches, shooting_data

def calculate_rolling_averages(data, stats, window=5):
    for stat in stats:
        # Beregn rolling averages for hjemme- og bortelag
        data[f'{stat}_home_rolling_avg'] = (
            data.groupby('home')[stat]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        data[f'{stat}_away_rolling_avg'] = (
            data.groupby('away')[stat]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return data

# Funksjon for å beregne resultatet for hver kamp
def calculate_result(score):
    try:
        home_score, away_score = map(int, score.split('–'))
        if home_score > away_score:
            return 1  # Hjemme-seier
        elif home_score == away_score:
            return 0  # Uavgjort
        else:
            return -1  # Borte-seier    
    except Exception as e:
        print(f"Feil med score formatet: {score} - {e}")
        return None

# Beregn rolling averages for resultater
def calculate_result_rolling_averages(data, window=5):
    result_home = data['score'].apply(lambda x: calculate_result(x) if pd.notnull(x) else None)
    data['result_home'] = result_home
    data['result_away'] = -result_home

    data['result_home_rolling_avg'] = (
        data.groupby('home')['result_home']
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    data['result_away_rolling_avg'] = (
        data.groupby('away')['result_away']
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return data

def calculate_match_results(matches):
    matches['result_home'] = matches['score'].apply(lambda x: calculate_result(x) if pd.notnull(x) else None)
    matches['result_away'] = matches['score'].apply(lambda x: calculate_result(x) if pd.notnull(x) else None)
    return matches

