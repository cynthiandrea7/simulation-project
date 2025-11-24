import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://www.slickcharts.com/sp500"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
resp = requests.get(url, headers=headers)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

# Find the table
table = soup.find("table", {"class": "table table-hover table-borderless table-sm"})
headers = [th.text.strip() for th in table.find("thead").find_all("th")]

rows = []
for tr in table.find("tbody").find_all("tr"):
    cells = [td.text.strip() for td in tr.find_all("td")]
    rows.append(cells)

df = pd.DataFrame(rows, columns=headers)

# Clean up columns: remove % from Weight, convert to numeric
df['Weight'] = df['Weight'].str.rstrip('%').astype(float) / 100.0

# Optional: convert Price, Chg, % Chg to numeric
df['Price'] = df['Price'].str.replace(',','').astype(float)
df['Chg'] = df['Chg'].str.replace(',','').astype(float)
df['% Chg'] = df['% Chg'].str.replace('%','').str.replace('(','-').str.replace(')','').str.replace('--','-').astype(float) / 100.0

# Save CSV
df.to_csv("sp500_components_slickcharts.csv", index=False)
print("Saved to sp500_components_slickcharts.csv")
