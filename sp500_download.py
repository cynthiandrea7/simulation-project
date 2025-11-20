import yfinance as yf
import pandas as pd
from tqdm import tqdm
import os

# Ensure output folder exists
os.makedirs("data", exist_ok=True)

# Load metadata (clean S&P500 tickers + sectors)
df_meta = pd.read_csv("sp500_sectors.csv").drop_duplicates(subset="Symbol")

tickers = df_meta["Symbol"].tolist()
print(f"Loaded {len(tickers)} tickers")

all_data = []

# ------------------------------------------
# STEP 1 — DOWNLOAD EACH TICKER CLEANLY
# ------------------------------------------
for ticker in tqdm(tickers, desc="Downloading price data"):
    try:
        df = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        # Skip empty or invalid downloads
        if df is None or df.empty:
            continue

        df = df.copy()
        df["Symbol"] = str(ticker).upper().strip()

        # Reset index (Date → column)
        df = df.reset_index()

        # Flatten multiindex columns if needed
        df.columns = [c if not isinstance(c, tuple) else c[0] for c in df.columns]

        # Keep standard OHLCV columns
        keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]
        df = df[[c for c in df.columns if c in keep_cols]]

        all_data.append(df)

    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        continue

# ------------------------------------------
# STEP 2 — CONCATENATE ALL DATA
# ------------------------------------------
df_prices = pd.concat(all_data, axis=0, ignore_index=True)

# Ensure Symbol formatting matches df_meta
df_prices["Symbol"] = df_prices["Symbol"].str.upper().str.strip()
df_meta["Symbol"] = df_meta["Symbol"].str.upper().str.strip()

# ------------------------------------------
# STEP 3 — MERGE SECTOR METADATA
# ------------------------------------------
df_final = df_prices.merge(df_meta, on="Symbol", how="left")

# ------------------------------------------
# STEP 4 — SAVE OUTPUT FILES
# ------------------------------------------
df_final.to_csv("data/sp500_5yr_with_sectors.csv", index=False)
df_final.to_parquet("data/sp500_5yr_with_sectors.parquet")

print("\nDONE! Saved:")
print(" - data/sp500_5yr_with_sectors.csv")
print(" - data/sp500_5yr_with_sectors.parquet")

print("\nPREVIEW:")
print(df_final.head())



