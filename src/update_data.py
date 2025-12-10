import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def update_single_ticker(ticker, raw_dir):
    path = Path(raw_dir) / f"{ticker}.csv"

    if path.exists():
        old = pd.read_csv(path)
        old["Date"] = pd.to_datetime(old["Date"])
        last_date = old["Date"].max()
        start = last_date.strftime("%Y-%m-%d")
    else:
        old = None
        start = "2010-01-01"

    new = yf.download(ticker, start=start)
    if new.empty:
        return

    new.reset_index(inplace=True)
    new["Ticker"] = ticker

    if old is not None:
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["Date"], keep="last")
    else:
        merged = new

    merged.to_csv(path, index=False)


def main():
    cfg = load_config()
    tickers = cfg["data"]["tickers"]
    raw_dir = cfg["paths"]["raw_dir"]
    Path(raw_dir).mkdir(exist_ok=True, parents=True)

    for t in tickers:
        update_single_ticker(t, raw_dir)

    print("Update complete.")
