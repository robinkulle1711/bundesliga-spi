"""
calculate_ratings.py — Phase 3, Step 9

Processes all historical matches in chronological order through the
Attack/Defense model and saves two outputs:

    data/processed/team_ratings_history.csv
        date, season, match_id, home_team, away_team,
        home_goals, away_goals, result,
        home_atk_pre, home_def_pre, away_atk_pre, away_def_pre,
        home_atk_post, home_def_post, away_atk_post, away_def_post

    data/processed/current_ratings.csv
        team, attack, defense, power
        (defense = raw def_ parameter; display as -defense for defensive_solidity)
"""

import pandas as pd
from pathlib import Path
from attack_defense_model import AttackDefenseRating
from model_config import AD_PARAMS

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    # ------------------------------------------------------------------ #
    # Load                                                                 #
    # ------------------------------------------------------------------ #
    matches_path = PROCESSED_DIR / "matches_clean.csv"
    if not matches_path.exists():
        raise FileNotFoundError(f"Run data_prep.py first — {matches_path} not found.")

    df = pd.read_csv(matches_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} matches ({df['date'].min().date()} to {df['date'].max().date()})")

    # ------------------------------------------------------------------ #
    # Process matches                                                      #
    # ------------------------------------------------------------------ #
    model = AttackDefenseRating(**AD_PARAMS)
    history = []

    for match_id, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])

        # Capture PRE-match ratings (initialise new teams at 0 for logging)
        pre_atk_h = model.get_attack(home)
        pre_def_h = model.get_defense(home)
        pre_atk_a = model.get_attack(away)
        pre_def_a = model.get_defense(away)

        post_atk_h, post_def_h, post_atk_a, post_def_a = model.update_rating(
            home, away, hg, ag,
            season=row["season"],
            match_date=row["date"].date(),
        )

        history.append({
            "date":         row["date"],
            "season":       row["season"],
            "match_id":     match_id,
            "home_team":    home,
            "away_team":    away,
            "home_goals":   hg,
            "away_goals":   ag,
            "result":       row["result"],
            # Pre-match ratings
            "home_atk_pre": round(pre_atk_h, 4),
            "home_def_pre": round(pre_def_h, 4),
            "away_atk_pre": round(pre_atk_a, 4),
            "away_def_pre": round(pre_def_a, 4),
            # Post-match ratings
            "home_atk_post": round(post_atk_h, 4),
            "home_def_post": round(post_def_h, 4),
            "away_atk_post": round(post_atk_a, 4),
            "away_def_post": round(post_def_a, 4),
        })

    history_df = pd.DataFrame(history)

    # ------------------------------------------------------------------ #
    # Current ratings snapshot                                             #
    # ------------------------------------------------------------------ #
    current_df = (
        pd.DataFrame(model.all_ratings())
        .sort_values("power", ascending=False)
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    history_path = PROCESSED_DIR / "team_ratings_history.csv"
    current_path = PROCESSED_DIR / "current_ratings.csv"

    history_df.to_csv(history_path, index=False)
    current_df.to_csv(current_path, index=False)

    print(f"Saved history  -> {history_path}  ({len(history_df)} rows)")
    print(f"Saved current  -> {current_path}  ({len(current_df)} teams)")

    return history_df, current_df


def print_summary(history_df: pd.DataFrame, current_df: pd.DataFrame) -> None:
    print("\n=== TOP 10 TEAMS (current power) ===")
    print(f"{'Rank':<5} {'Team':<20} {'Attack':>8} {'Defense':>8} {'Power':>8}")
    for i, row in current_df.head(10).iterrows():
        # Display defensive_solidity = -defense (higher = better)
        ds = -row["defense"]
        print(
            f"  {i+1:<4} {row['team']:<20} "
            f"{row['attack']:>+8.3f} "
            f"{ds:>+8.3f} "
            f"{row['power']:>+8.3f}"
        )

    print("\n=== FULL RATINGS TABLE (attack | def_solidity | power) ===")
    print(f"{'Team':<20} {'Attack':>8} {'Def Sol':>8} {'Power':>8}")
    for _, row in current_df.iterrows():
        bar = "#" * max(0, int(row["power"] * 10 + 5))
        print(
            f"  {row['team']:<20} "
            f"{row['attack']:>+8.3f} "
            f"{-row['defense']:>+8.3f} "
            f"{row['power']:>+8.3f}  {bar}"
        )

    print(f"\n=== RATING SPREAD ===")
    top = current_df.iloc[0]
    bot = current_df.iloc[-1]
    print(f"  Highest power: {top['team']} — {top['power']:+.3f}")
    print(f"  Lowest  power: {bot['team']} — {bot['power']:+.3f}")
    print(f"  Spread: {top['power'] - bot['power']:.3f}")

    print(f"\n=== ATTACK LEADERS ===")
    for _, r in current_df.nlargest(3, "attack").iterrows():
        print(f"    {r['team']:<20} atk={r['attack']:+.3f}")
    print(f"\n=== BEST DEFENSES (by defensive solidity = -def_) ===")
    for _, r in current_df.nsmallest(3, "defense").iterrows():
        print(f"    {r['team']:<20} def_sol={-r['defense']:+.3f}")

    # Sanity: check parameter magnitudes
    max_atk = current_df["attack"].abs().max()
    max_def = current_df["defense"].abs().max()
    print(f"\n=== PARAMETER STABILITY ===")
    print(f"  Max |attack| = {max_atk:.3f}  (should be < 2.0)")
    print(f"  Max |defense| = {max_def:.3f}  (should be < 2.0)")


if __name__ == "__main__":
    history_df, current_df = run()
    print_summary(history_df, current_df)
