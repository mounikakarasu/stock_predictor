import sys
import pandas as pd
import joblib
import yaml
from pathlib import Path


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    ticker = sys.argv[1].upper()

    cfg = load_config()
    df = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "train_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    models_dir = Path(cfg["paths"]["models_dir"])
    model = joblib.load(models_dir / "model.pkl")
    le = joblib.load(models_dir / "label_encoder.pkl")

    scaler_path = models_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    feature_cols = joblib.load(models_dir / "features.pkl")

    sub = df[df["Ticker"] == ticker]
    if sub.empty:
        sys.exit(1)

    latest = sub.sort_values("Date").tail(1)
    X = latest[feature_cols].values

    if scaler is not None:
        X = scaler.transform(X)

    pred = model.predict(X)
    label = le.inverse_transform(pred)[0]

    print(label)


if __name__ == "__main__":
    main()
