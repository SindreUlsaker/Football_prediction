import pandas as pd
from data_fetching import main as fetch_data_main
from data_processing import process_matches
from model_training import train_poisson_models
from prediction import predict_poisson_from_models


def main(round_numbers: list[int]):
    # 1) Hent rådata (valgfritt)
    # fetch_data_main()

    # 2) Les kampdata
    df_all = pd.read_csv("premier_league_matches_full.csv", parse_dates=["date"])

    # 3) Definer rolling-vinduer per stat
    stat_windows = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}

    # 4) Kjør full datapipeline: preprosess + features
    df = process_matches(df_all, stat_windows)
    df.to_csv("processed_matches.csv", index=False)
    print("Lagrer processed_matches.csv")

    # 5) Bygg feature-lister
    # Hjemmelag: kun offensive hjemmestats + defensive bortestats
    features_home = (
        [
            f"{stat}_home_roll{w}"
            for stat in ("xg", "gf")
            for w in stat_windows[stat]
        ]
        + [f"xg_conceded_away_roll{w}" for w in stat_windows["xg"]]
        + [f"ga_away_roll{w}" for w in stat_windows["ga"]]
        + ["avg_goals_for_home", "avg_goals_against_away"]
    )

    # Bortelag: kun offensive bortestats + defensive hjemmestats
    features_away = (
        [
            f"{stat}_away_roll{w}"
            for stat in ("xg", "gf")
            for w in stat_windows[stat]
        ]
        + [f"xg_conceded_home_roll{w}" for w in stat_windows["xg"]]
        + [f"ga_home_roll{w}" for w in stat_windows["ga"]]
        + ["avg_goals_for_away", "avg_goals_against_home"]
    )

    # 6) Tren Poisson-modeller
    model_home, model_away, scaler_home, scaler_away = train_poisson_models(df, features_home, features_away)
    print("Trente Poisson-regresjonsmodeller for mål")

    # 7) Hent fremtidige kamper
    future = df[df["round"].isin(round_numbers) & df["result_home"].isna()].copy()
    if future.empty:
        print("Ingen kamper funnet for valgte runder.")
        return

    # 8) Prediker sannsynligheter via Poisson
    result_df = predict_poisson_from_models(
        model_home=model_home,
        model_away=model_away,
        scaler_home=scaler_home,
        scaler_away=scaler_away,
        df_future=future,
        features_home=features_home,
        features_away=features_away,
        max_goals=10,
    )

    # 9) Vis resultat
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main(round_numbers=[37, 38, 39])
