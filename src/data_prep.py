"""
data_prep.py — Phase 2, Step 7

Loads all raw Bundesliga CSVs from data/raw/, standardizes team names,
and produces a clean match dataframe saved to data/processed/matches_clean.csv.

Output columns:
    date         : match date (datetime)
    season       : e.g. "2324"
    home_team    : standardized home team name
    away_team    : standardized away team name
    home_goals   : full-time home goals (int)
    away_goals   : full-time away goals (int)
    result       : "H" / "D" / "A"
    home_shots   : shots (optional, kept if present)
    away_shots   : shots (optional, kept if present)
    home_shots_target : shots on target (optional)
    away_shots_target : shots on target (optional)
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Map raw names → canonical names.
# Current names are already clean; this dict is here for future-proofing
# (e.g. if a promoted club appears under a different spelling).
TEAM_NAME_MAP: dict[str, str] = {
    # Add overrides here if needed, e.g.:
    # "FC Bayern Muenchen": "Bayern Munich",
}


def _season_from_filename(stem: str) -> str:
    """Extract season code from filename stem, e.g. 'D1_2324' → '2324'."""
    return stem.split("_", 1)[-1]


def load_raw_files() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("D1_*.csv"))
    if not files:
        raise FileNotFoundError(f"No D1_*.csv files found in {RAW_DIR}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="latin1")

        # Drop BOM duplicate column if present (﻿Div)
        df.columns = [c.lstrip("\ufeff") for c in df.columns]

        df["season"] = _season_from_filename(f.stem)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # defragment before adding columns
    # Parse dates (format is DD/MM/YY in these files)
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Drop rows where the essential fields are missing
    df = df.dropna(subset=["date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    # Standardize team names
    df["home_team"] = df["HomeTeam"].str.strip().replace(TEAM_NAME_MAP)
    df["away_team"] = df["AwayTeam"].str.strip().replace(TEAM_NAME_MAP)

    # Core columns
    df["home_goals"] = df["FTHG"].astype(int)
    df["away_goals"] = df["FTAG"].astype(int)
    df["result"] = df["FTR"]  # H / D / A

    # Optional columns (shots, shots on target) — build all at once to avoid fragmentation
    optional = {"HS": "home_shots", "AS": "away_shots",
                "HST": "home_shots_target", "AST": "away_shots_target"}
    extra = {dst: pd.to_numeric(df[src], errors="coerce")
             for src, dst in optional.items() if src in df.columns}

    core = df[["date", "season", "home_team", "away_team",
               "home_goals", "away_goals", "result"]].copy()

    if extra:
        core = pd.concat([core, pd.DataFrame(extra, index=df.index)], axis=1)

    return core.sort_values("date").reset_index(drop=True)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw files...")
    raw = load_raw_files()
    print(f"  {len(raw)} rows loaded from {RAW_DIR}")

    print("Cleaning data...")
    clean_df = clean(raw)
    print(f"  {len(clean_df)} rows after cleaning")

    out_path = PROCESSED_DIR / "matches_clean.csv"
    clean_df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    # Summary
    print(f"\nSeasons:  {sorted(clean_df['season'].unique())}")
    print(f"Date range: {clean_df['date'].min().date()} to {clean_df['date'].max().date()}")
    print(f"Teams ({clean_df['home_team'].nunique()}): "
          f"{sorted(clean_df['home_team'].unique())}")
    res = clean_df["result"].value_counts(normalize=True).mul(100).round(1)
    print(f"Results: H={res.get('H',0)}%  D={res.get('D',0)}%  A={res.get('A',0)}%")


if __name__ == "__main__":
    main()
