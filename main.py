import pandas as pd
from data_fetching_new import main as fetch_data_main
from data_processing_new import process_matches
from model_training import train_model, scaler
from prediction import predict_future_matches


def main(round_numbers: list[int]):
    # fetch_data_main()  # om du vil laste helt nytt

    # 1) Les rådata
    df_all = pd.read_csv("premier_league_matches_full.csv", parse_dates=["date"])

    # 2) Kjør full prosessering + feature‐engineering
    rolling_stats = ["xg_home", "xg_away", "gf_home", "gf_away"]
    rolling_windows = [5, 10]
    df = process_matches(df_all, rolling_stats, rolling_windows)

    # 3) Lagre (valgfritt)
    df.to_csv("processed_matches.csv", index=False)
    print("Saved processed_matches.csv")

    # 4) Definer features og target
    dynamic = [f"{s}_roll{w}" for s in rolling_stats for w in rolling_windows]
    static = [
        "avg_points_home",
        "avg_points_away",
        "avg_goals_for_home",
        "avg_goals_for_away",
        "avg_goals_against_home",
        "avg_goals_against_away",
        "home_advantage",
    ]
    features = dynamic + static
    target = "result_home"

    # 5) Tren modellen
    model = train_model(df, features, target)
    print(f"Model trained on {len(features)} features.")

    # 6) Plukk ut fremtidige kamper
    future = df[df["round"].isin(round_numbers) & df[target].isna()].reset_index(
        drop=True
    )

    if future.empty:
        print(f"No future matches for rounds {round_numbers}")
        return

    # 7) Bruk predict_future_matches fra prediction.py
    result_df = predict_future_matches(
        model=model, scaler=scaler, future_matches=future, features=features
    )

    # 8) Vis tabell
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main(round_numbers=[37, 38])
