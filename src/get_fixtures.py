"""
get_fixtures.py — Phase 4, Step 12

Downloads the current 2025-26 Bundesliga season data and saves it to
data/raw/D1_2526.csv (completed matches only — football-data.co.uk does not
publish future fixtures).

Also writes a sample upcoming fixtures file to data/raw/upcoming_fixtures.csv
for use by predict_matches.py.  Update this file with the actual schedule
before each matchday.  Columns: date, matchday, home_team, away_team.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import date

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

CURRENT_SEASON = "2526"
CURRENT_SEASON_URL = (
    f"https://www.football-data.co.uk/mmz4281/{CURRENT_SEASON}/D1.csv"
)

# ---------------------------------------------------------------------------
# Matchday 25 fixtures (week of 28 Feb – 2 Mar 2026).
# Replace with actual schedule when known.
# ---------------------------------------------------------------------------
UPCOMING_FIXTURES = [
    {"date": "2026-02-28", "matchday": 25, "home_team": "Bayern Munich",  "away_team": "Hoffenheim"},
    {"date": "2026-02-28", "matchday": 25, "home_team": "Dortmund",       "away_team": "Leverkusen"},
    {"date": "2026-03-01", "matchday": 25, "home_team": "Stuttgart",      "away_team": "RB Leipzig"},
    {"date": "2026-03-01", "matchday": 25, "home_team": "Ein Frankfurt",  "away_team": "Werder Bremen"},
    {"date": "2026-03-01", "matchday": 25, "home_team": "Augsburg",       "away_team": "FC Koln"},
    {"date": "2026-03-01", "matchday": 25, "home_team": "Hamburg",        "away_team": "Mainz"},
    {"date": "2026-03-01", "matchday": 25, "home_team": "Freiburg",       "away_team": "Union Berlin"},
    {"date": "2026-03-02", "matchday": 25, "home_team": "M'gladbach",     "away_team": "Wolfsburg"},
    {"date": "2026-03-02", "matchday": 25, "home_team": "St Pauli",       "away_team": "Heidenheim"},
]


def download_current_season() -> bool:
    dest = RAW_DIR / f"D1_{CURRENT_SEASON}.csv"
    print(f"Downloading {CURRENT_SEASON} season ...", end=" ", flush=True)
    try:
        r = requests.get(CURRENT_SEASON_URL, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        rows = len(r.text.strip().splitlines()) - 1
        print(f"OK ({rows} completed matches)")
        return True
    except requests.RequestException as e:
        print(f"FAILED - {e}")
        return False


def write_upcoming_fixtures() -> None:
    dest = RAW_DIR / "upcoming_fixtures.csv"
    df = pd.DataFrame(UPCOMING_FIXTURES)
    df.to_csv(dest, index=False)
    print(f"Upcoming fixtures written -> {dest}  ({len(df)} matches, matchday {df['matchday'].iloc[0]})")
    print("\n  Note: edit data/raw/upcoming_fixtures.csv to match the actual schedule.")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_current_season()
    write_upcoming_fixtures()


if __name__ == "__main__":
    main()
