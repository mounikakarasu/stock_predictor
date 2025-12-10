import streamlit as st
import pandas as pd
import joblib
import yaml
from pathlib import Path


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


cfg = load_config()

df = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "train_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])

models_dir = Path(cfg["paths"]["models_dir"])
model = joblib.load(models_dir / "model.pkl")
label_encoder = joblib.load(models_dir / "label_encoder.pkl")

scaler_path = models_dir / "scaler.pkl"
scaler = joblib.load(scaler_path) if scaler_path.exists() else None

feature_cols = joblib.load(models_dir / "features.pkl")

tickers = sorted(df["Ticker"].unique())


def get_latest(ticker):
    sub = df[df["Ticker"] == ticker]
    if sub.empty:
        return None
    latest = sub.sort_values("Date").tail(1)
    X = latest[feature_cols].values
    if scaler:
        X = scaler.transform(X)
    pred = model.predict(X)
    label = label_encoder.inverse_transform(pred)[0]
    return latest.iloc[0]["Date"], label


st.title("Stock Trend Prediction Viewer")

choice = st.selectbox("Select Ticker", tickers)

date, label = get_latest(choice)
st.write("Date:", date)
st.write("Prediction:", label)
