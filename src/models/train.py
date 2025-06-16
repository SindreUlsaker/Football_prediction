from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd


def train_poisson_models(
    data: pd.DataFrame, features_home: list[str], features_away: list[str]
):
    """
    Trener Poisson-modeller for gf_home og gf_away med egne scalers.
    """

    df = data.dropna(subset=features_home + features_away + ["gf_home", "gf_away"])

    # Hjemmelag
    Xh = df[features_home].copy()
    Xh["is_home"] = 1
    yh = df["gf_home"]

    # Bortelag
    Xa = df[features_away].copy()
    Xa["is_home"] = 0
    ya = df["gf_away"]

    # Én scaler for hver
    scaler_home = StandardScaler().fit(Xh)
    scaler_away = StandardScaler().fit(Xa)

    Xh_s = scaler_home.transform(Xh)
    Xa_s = scaler_away.transform(Xa)

    model_home = PoissonRegressor(alpha=1.0, max_iter=300).fit(Xh_s, yh)
    model_away = PoissonRegressor(alpha=1.0, max_iter=300).fit(Xa_s, ya)

    return model_home, model_away, scaler_home, scaler_away
