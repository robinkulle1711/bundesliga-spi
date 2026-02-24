"""
predict_matches.py — Phase 3, Step 10

Replays ALL historical matches through both the Attack/Defense and Elo models,
then generates predictions for every upcoming fixture.

Output: data/predictions/upcoming_predictions.csv
Columns:
    date, matchday, home_team, away_team,
    --- AD model ---
    home_attack, home_defense, away_attack, away_defense,
    home_power, away_power,
    home_win_prob, draw_prob, away_win_prob,
    xg_home, xg_away,
    predicted_home_goals, predicted_away_goals, most_likely_result,
    --- Elo model ---
    elo_home_rating, elo_away_rating,
    elo_home_win_prob, elo_draw_prob, elo_away_win_prob,
    elo_xg_home, elo_xg_away,
    elo_most_likely_result
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from attack_defense_model import AttackDefenseRating
from elo_model import EloRating
from model_config import AD_PARAMS, ELO_PARAMS
from data_prep import load_raw_files, clean

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PRED_DIR      = Path(__file__).parent.parent / "data" / "predictions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_completed_matches() -> pd.DataFrame:
    raw = load_raw_files()
    df  = clean(raw)
    return df.sort_values("date").reset_index(drop=True)


def build_ad_ratings(matches: pd.DataFrame) -> AttackDefenseRating:
    model = AttackDefenseRating(**AD_PARAMS)
    for _, row in matches.iterrows():
        model.update_rating(
            row["home_team"], row["away_team"],
            int(row["home_goals"]), int(row["away_goals"]),
            season=row["season"],
            match_date=row["date"].date(),
        )
    return model


def build_elo_ratings(matches: pd.DataFrame) -> EloRating:
    model    = EloRating(**ELO_PARAMS)
    ref_date = matches["date"].max()
    for _, row in matches.iterrows():
        model.update_rating(
            row["home_team"], row["away_team"],
            int(row["home_goals"]), int(row["away_goals"]),
            season=row["season"],
            match_date=row["date"].date(),
            reference_date=ref_date.date(),
        )
    return model


def load_fixtures() -> pd.DataFrame:
    path = RAW_DIR / "upcoming_fixtures.csv"
    if not path.exists():
        raise FileNotFoundError(f"No fixtures file at {path}. Run get_fixtures.py first.")
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def most_likely(p_home: float, p_draw: float, p_away: float) -> str:
    m = max(p_home, p_draw, p_away)
    return "H" if m == p_home else ("D" if m == p_draw else "A")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading completed matches ...")
    matches = load_all_completed_matches()
    seasons = sorted(matches["season"].unique())
    print(f"  {len(matches)} matches across seasons: {seasons}")

    print("\nBuilding Attack/Defense ratings ...")
    ad_model = build_ad_ratings(matches)

    print("Building Elo ratings ...")
    elo_model = build_elo_ratings(matches)

    print("\nLoading upcoming fixtures ...")
    fixtures = load_fixtures()
    print(f"  {len(fixtures)} fixtures (matchday {fixtures['matchday'].iloc[0]})")

    print("\nGenerating predictions ...")
    rows = []
    for _, fix in fixtures.iterrows():
        home, away = fix["home_team"], fix["away_team"]
        ad  = ad_model.predict_outcome(home, away)
        elo = elo_model.predict_outcome(home, away)

        rows.append({
            "date":                   fix["date"].date(),
            "matchday":               int(fix["matchday"]),
            "home_team":              home,
            "away_team":              away,
            # AD
            "home_attack":            ad["home_attack"],
            "home_defense":           ad["home_defense"],
            "away_attack":            ad["away_attack"],
            "away_defense":           ad["away_defense"],
            "home_power":             ad["home_power"],
            "away_power":             ad["away_power"],
            "home_win_prob":          ad["home_win_prob"],
            "draw_prob":              ad["draw_prob"],
            "away_win_prob":          ad["away_win_prob"],
            "xg_home":                ad["xg_home"],
            "xg_away":                ad["xg_away"],
            "predicted_home_goals":   round(ad["xg_home"]),
            "predicted_away_goals":   round(ad["xg_away"]),
            "most_likely_result":     most_likely(ad["home_win_prob"], ad["draw_prob"], ad["away_win_prob"]),
            # Elo
            "elo_home_rating":        elo["home_rating"],
            "elo_away_rating":        elo["away_rating"],
            "elo_home_win_prob":      elo["home_win_prob"],
            "elo_draw_prob":          elo["draw_prob"],
            "elo_away_win_prob":      elo["away_win_prob"],
            "elo_xg_home":            elo["xg_home"],
            "elo_xg_away":            elo["xg_away"],
            "elo_most_likely_result": most_likely(elo["home_win_prob"], elo["draw_prob"], elo["away_win_prob"]),
        })

    pred_df = pd.DataFrame(rows)
    out_path = PRED_DIR / "upcoming_predictions.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    _print_table(pred_df)


def _print_table(df: pd.DataFrame) -> None:
    sep = "-" * 105
    for label, hw, dw, aw, xh, xa, tip in [
        ("ATTACK/DEFENSE", "home_win_prob", "draw_prob", "away_win_prob",
         "xg_home", "xg_away", "most_likely_result"),
        ("ELO", "elo_home_win_prob", "elo_draw_prob", "elo_away_win_prob",
         "elo_xg_home", "elo_xg_away", "elo_most_likely_result"),
    ]:
        print(f"\n{'=' * 105}")
        print(f"  MATCHDAY {df['matchday'].iloc[0]} — {label} MODEL")
        print(f"{'=' * 105}")
        print(f"  {'Date':<12} {'Home':<18} {'Away':<18} {'H%':>5} {'D%':>5} {'A%':>5}  {'xG':>8}  {'Tip':>4}")
        print(sep)
        for _, r in df.iterrows():
            print(
                f"  {str(r['date']):<12} {r['home_team']:<18} {r['away_team']:<18} "
                f"{r[hw]*100:>4.0f}% {r[dw]*100:>4.0f}% {r[aw]*100:>4.0f}%  "
                f"{r[xh]:.1f}-{r[xa]:.1f}  {r[tip]:>4}"
            )
        print(sep)


if __name__ == "__main__":
    main()
