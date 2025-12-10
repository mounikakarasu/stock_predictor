from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import yaml
from pathlib import Path

app = FastAPI(title="Stock Prediction API")


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


cfg = load_config()
processed_path = Path(cfg["paths"]["processed_dir"]) / "train_dataset.csv"
df = pd.read_csv(processed_path)
df["Date"] = pd.to_datetime(df["Date"])

models_dir = Path(cfg["paths"]["models_dir"])
model = joblib.load(models_dir / "model.pkl")
label_encoder = joblib.load(models_dir / "label_encoder.pkl")

scaler_path = models_dir / "scaler.pkl"
scaler = joblib.load(scaler_path) if scaler_path.exists() else None

feature_cols = joblib.load(models_dir / "features.pkl")


def get_latest_features(ticker: str):
    sub = df[df["Ticker"] == ticker]

    if sub.empty:
        return None, None

    latest = sub.sort_values("Date").tail(1)
    X = latest[feature_cols].values

    if scaler is not None:
        X = scaler.transform(X)

    return latest, X


@app.get("/predict/{ticker}")
def predict_ticker(ticker: str):
    ticker = ticker.upper()

    latest, X = get_latest_features(ticker)

    if latest is None:
        raise HTTPException(status_code=404, detail="Ticker not found.")

    pred_encoded = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return {
        "Ticker": ticker,
        "Date": str(latest["Date"].iloc[0].date()),
        "Prediction": pred_label,
    }
