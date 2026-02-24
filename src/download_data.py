"""
download_data.py — Phase 2, Step 5

Downloads the last 5 seasons of Bundesliga 1 data from football-data.co.uk.
URL pattern: https://www.football-data.co.uk/mmz4281/YYZZ/D1.csv
  YYZZ = last two digits of start year + last two digits of end year
  e.g. 2024-25 season → "2425"

Saves files to data/raw/D1_YYZZ.csv
"""

import requests
import time
from pathlib import Path

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/D1.csv"

# Last 5 completed or in-progress seasons (most recent first)
SEASONS = ["2425", "2324", "2223", "2122", "2021"]

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_season(season: str) -> bool:
    url = BASE_URL.format(season=season)
    dest = RAW_DIR / f"D1_{season}.csv"

    if dest.exists():
        print(f"  {season}: already downloaded, skipping.")
        return True

    print(f"  {season}: downloading from {url} ...", end=" ", flush=True)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)
        rows = len(response.text.strip().splitlines()) - 1  # subtract header
        print(f"OK ({rows} matches, {len(response.content):,} bytes)")
        return True
    except requests.HTTPError as e:
        print(f"FAILED — HTTP {e.response.status_code}")
        return False
    except requests.RequestException as e:
        print(f"FAILED — {e}")
        return False


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Bundesliga data to {RAW_DIR}\n")

    results = {}
    for season in SEASONS:
        results[season] = download_season(season)
        time.sleep(0.5)  # polite delay between requests

    print()
    ok = [s for s, v in results.items() if v]
    fail = [s for s, v in results.items() if not v]
    print(f"Done: {len(ok)} downloaded, {len(fail)} failed.")
    if fail:
        print(f"Failed seasons: {', '.join(fail)}")


if __name__ == "__main__":
    main()
