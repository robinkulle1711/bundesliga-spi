"""
calculate_ratings.py — Phase 3, Step 9

Runs both the Attack/Defense model and the Elo model over all historical
matches and saves two outputs:

    data/processed/team_ratings_history.csv
        date, season, match_id, home_team, away_team,
        home_goals, away_goals, result,
        home_atk_pre, home_def_pre, away_atk_pre, away_def_pre,
        home_atk_post, home_def_post, away_atk_post, away_def_post,
        home_elo_pre, away_elo_pre, home_elo_post, away_elo_post

    data/processed/current_ratings.csv
        team, attack, defense, power, elo_rating
"""

import pandas as pd
from pathlib import Path
from attack_defense_model import AttackDefenseRating
from elo_model import EloRating
from model_config import AD_PARAMS, ELO_PARAMS

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
    # Process matches through both models                                  #
    # ------------------------------------------------------------------ #
    ad_model  = AttackDefenseRating(**AD_PARAMS)
    elo_model = EloRating(**ELO_PARAMS)
    history   = []
    ref_date  = df["date"].max()

    for match_id, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag     = int(row["home_goals"]), int(row["away_goals"])

        # Pre-match ratings
        pre_atk_h  = ad_model.get_attack(home)
        pre_def_h  = ad_model.get_defense(home)
        pre_atk_a  = ad_model.get_attack(away)
        pre_def_a  = ad_model.get_defense(away)
        pre_elo_h  = elo_model.get_rating(home)
        pre_elo_a  = elo_model.get_rating(away)

        # Update both models
        post_atk_h, post_def_h, post_atk_a, post_def_a = ad_model.update_rating(
            home, away, hg, ag,
            season=row["season"],
            match_date=row["date"].date(),
        )
        post_elo_h, post_elo_a = elo_model.update_rating(
            home, away, hg, ag,
            season=row["season"],
            match_date=row["date"].date(),
            reference_date=ref_date.date(),
        )

        history.append({
            "date":           row["date"],
            "season":         row["season"],
            "match_id":       match_id,
            "home_team":      home,
            "away_team":      away,
            "home_goals":     hg,
            "away_goals":     ag,
            "result":         row["result"],
            # AD pre
            "home_atk_pre":   round(pre_atk_h, 4),
            "home_def_pre":   round(pre_def_h, 4),
            "away_atk_pre":   round(pre_atk_a, 4),
            "away_def_pre":   round(pre_def_a, 4),
            # AD post
            "home_atk_post":  round(post_atk_h, 4),
            "home_def_post":  round(post_def_h, 4),
            "away_atk_post":  round(post_atk_a, 4),
            "away_def_post":  round(post_def_a, 4),
            # Elo pre / post
            "home_elo_pre":   round(pre_elo_h, 2),
            "away_elo_pre":   round(pre_elo_a, 2),
            "home_elo_post":  round(post_elo_h, 2),
            "away_elo_post":  round(post_elo_a, 2),
        })

    history_df = pd.DataFrame(history)

    # ------------------------------------------------------------------ #
    # Current ratings snapshot (AD sorted by power + Elo rating added)    #
    # ------------------------------------------------------------------ #
    current_df = (
        pd.DataFrame(ad_model.all_ratings())
        .sort_values("power", ascending=False)
        .reset_index(drop=True)
    )
    elo_map = {t: round(r, 1) for t, r in elo_model.ratings.items()}
    current_df["elo_rating"] = current_df["team"].map(elo_map)

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
    print("\n=== TOP 10 TEAMS (AD power | Elo rating) ===")
    print(f"{'Rank':<5} {'Team':<20} {'Attack':>8} {'Def Sol':>8} {'Power':>8} {'Elo':>7}")
    for i, row in current_df.head(10).iterrows():
        print(
            f"  {i+1:<4} {row['team']:<20} "
            f"{row['attack']:>+8.3f} "
            f"{-row['defense']:>+8.3f} "
            f"{row['power']:>+8.3f} "
            f"{row['elo_rating']:>7.0f}"
        )


if __name__ == "__main__":
    history_df, current_df = run()
    print_summary(history_df, current_df)
