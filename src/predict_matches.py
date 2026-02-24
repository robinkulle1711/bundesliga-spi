"""
predict_matches.py — Phase 3, Step 10

Replays ALL historical matches (including the current 2025-26 season) through
the Attack/Defense model to produce up-to-date ratings, then generates
predictions for every fixture listed in data/raw/upcoming_fixtures.csv.

Output: data/predictions/upcoming_predictions.csv
Columns:
    date, matchday, home_team, away_team,
    home_attack, home_defense, away_attack, away_defense,
    home_power, away_power,
    home_win_prob, draw_prob, away_win_prob,
    xg_home, xg_away,
    predicted_home_goals, predicted_away_goals,
    most_likely_result
"""

import sys
import pandas as pd
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))
from attack_defense_model import AttackDefenseRating
from model_config import AD_PARAMS
from data_prep import load_raw_files, clean

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PRED_DIR      = Path(__file__).parent.parent / "data" / "predictions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_completed_matches() -> pd.DataFrame:
    """Load and clean all seasons including current (2526) if available."""
    raw = load_raw_files()
    df  = clean(raw)
    df  = df.sort_values("date").reset_index(drop=True)
    return df


def build_current_ratings(matches: pd.DataFrame) -> AttackDefenseRating:
    """Replay all matches chronologically to get today's AD ratings."""
    model = AttackDefenseRating(**AD_PARAMS)
    for _, row in matches.iterrows():
        model.update_rating(
            row["home_team"], row["away_team"],
            int(row["home_goals"]), int(row["away_goals"]),
            season=row["season"],
            match_date=row["date"].date(),
        )
    return model


def load_fixtures() -> pd.DataFrame:
    path = RAW_DIR / "upcoming_fixtures.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No fixtures file found at {path}. Run get_fixtures.py first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def most_likely_result(p_home: float, p_draw: float, p_away: float) -> str:
    m = max(p_home, p_draw, p_away)
    if m == p_home:
        return "H"
    if m == p_draw:
        return "D"
    return "A"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load all completed matches and replay through AD model
    print("Loading completed matches ...")
    matches = load_all_completed_matches()
    seasons = sorted(matches["season"].unique())
    print(f"  {len(matches)} matches across seasons: {seasons}")
    print(f"  Date range: {matches['date'].min().date()} to {matches['date'].max().date()}")

    print("\nBuilding current Attack/Defense ratings ...")
    model = build_current_ratings(matches)

    # 2. Load upcoming fixtures
    print("\nLoading upcoming fixtures ...")
    fixtures = load_fixtures()
    print(f"  {len(fixtures)} fixtures (matchday {fixtures['matchday'].iloc[0]})")

    # 3. Generate predictions
    print("\nGenerating predictions ...")
    rows = []
    for _, fix in fixtures.iterrows():
        pred = model.predict_outcome(fix["home_team"], fix["away_team"])
        rows.append({
            "date":                  fix["date"].date(),
            "matchday":              int(fix["matchday"]),
            "home_team":             fix["home_team"],
            "away_team":             fix["away_team"],
            "home_attack":           pred["home_attack"],
            "home_defense":          pred["home_defense"],
            "away_attack":           pred["away_attack"],
            "away_defense":          pred["away_defense"],
            "home_power":            pred["home_power"],
            "away_power":            pred["away_power"],
            "home_win_prob":         pred["home_win_prob"],
            "draw_prob":             pred["draw_prob"],
            "away_win_prob":         pred["away_win_prob"],
            "xg_home":               pred["xg_home"],
            "xg_away":               pred["xg_away"],
            "predicted_home_goals":  round(pred["xg_home"]),
            "predicted_away_goals":  round(pred["xg_away"]),
            "most_likely_result":    most_likely_result(
                pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]
            ),
        })

    pred_df = pd.DataFrame(rows)

    # 4. Save
    out_path = PRED_DIR / "upcoming_predictions.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    # 5. Print formatted table
    _print_table(pred_df, model)


def _print_table(df: pd.DataFrame, model: AttackDefenseRating) -> None:
    sep = "-" * 100
    print(f"\n{'=' * 100}")
    print(f"  BUNDESLIGA MATCHDAY {df['matchday'].iloc[0]} PREDICTIONS  (Attack/Defense Model)")
    print(f"{'=' * 100}")
    print(
        f"  {'Date':<12} {'Home':<18} {'Away':<18} "
        f"{'H%':>5} {'D%':>5} {'A%':>5}  "
        f"{'xG':>8}  {'Pred':>6}  {'Tip':>4}"
    )
    print(sep)

    for _, r in df.iterrows():
        xg_str   = f"{r['xg_home']:.1f}-{r['xg_away']:.1f}"
        pred_str = f"{r['predicted_home_goals']:.0f}-{r['predicted_away_goals']:.0f}"
        print(
            f"  {str(r['date']):<12} "
            f"{r['home_team']:<18} "
            f"{r['away_team']:<18} "
            f"{r['home_win_prob']*100:>4.0f}% "
            f"{r['draw_prob']*100:>4.0f}% "
            f"{r['away_win_prob']*100:>4.0f}%  "
            f"{xg_str:>8}  "
            f"{pred_str:>6}  "
            f"{r['most_likely_result']:>4}"
        )

    print(sep)

    print(f"\n  Current top 5 by power:")
    for i, r in enumerate(model.all_ratings()[:5], 1):
        print(
            f"    {i}. {r['team']:<20} "
            f"atk={r['attack']:+.3f}  "
            f"def_sol={-r['defense']:+.3f}  "
            f"power={r['power']:+.3f}"
        )


if __name__ == "__main__":
    main()
