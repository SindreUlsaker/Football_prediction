from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

# Globale variabler for skalering (brukes også i prediction senere)
scaler = StandardScaler()


def train_model(data, features, target):
    train_data = data.dropna(
        subset=features + [target]
    )  # fjern kamper uten target (fremtidige)
    X = train_data[features]
    # Cast target til int for diskré klasser
    y = train_data[target].astype(int)

    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=42)
    model.fit(X_scaled, y)
    return model


def evaluate_model(model, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    loss = log_loss(y_test, proba)
    return {"accuracy": acc, "log_loss": loss}
