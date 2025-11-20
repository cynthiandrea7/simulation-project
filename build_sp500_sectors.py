import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def build_sp500_sectors(output_path: str = "sp500_sectors.csv"):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # Download webpage
    response = requests.get(URL, headers=headers)
    response.raise_for_status()

    # Read all tables from page
    tables = pd.read_html(response.text)

    # Table 1 contains the S&P 500 constituents on your system
    df = tables[1]

    # Rename the columns we want
    df = df.rename(columns={
        "GICS Sector": "Sector"
    })

    # Keep only needed columns
    df = df[["Symbol", "Security", "Sector"]]

    # Fix BRK.B -> BRK-B (Yahoo Finance format)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False).str.strip()

    # Remove duplicates
    df = df.drop_duplicates(subset="Symbol").reset_index(drop=True)

    # Save output
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    build_sp500_sectors()
