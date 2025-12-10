import pandas as pd
import yaml
import joblib
from pathlib import Path


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_artifacts(cfg):
    models_dir = Path(cfg["paths"]["models_dir"])
    model = joblib.load(models_dir / "model.pkl")
    le = joblib.load(models_dir / "label_encoder.pkl")

    scaler_path = models_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    feature_cols = joblib.load(models_dir / "features.pkl")

    return model, le, scaler, feature_cols


def load_dataset(cfg):
    df = pd.read_csv(Path(cfg["paths"]["processed_dir"]) / "train_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def prepare_latest_rows(df, feature_cols, scaler):
    latest = df.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(1)
    X = latest[feature_cols].values

    if scaler is not None:
        X = scaler.transform(X)

    return latest, X


def main():
    cfg = load_config()
    df = load_dataset(cfg)

    model, le, scaler, feature_cols = load_artifacts(cfg)

    latest, X = prepare_latest_rows(df, feature_cols, scaler)

    pred = model.predict(X)
    labels = le.inverse_transform(pred)

    out = latest[["Ticker", "Date"]].copy()
    out["Prediction"] = labels

    output_path = Path(cfg["paths"]["processed_dir"]) / "latest_predictions.csv"
    out.to_csv(output_path, index=False)

    print(out)


if __name__ == "__main__":
    main()
