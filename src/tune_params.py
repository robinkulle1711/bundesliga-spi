"""
tune_params.py — Parameter Grid Search

Finds optimal parameters by minimising Brier score across all historical
matches using proper forward-chaining (pre-match ratings predict each match).

Usage:
    python tune_params.py --model elo   # Elo grid search (default)
    python tune_params.py --model ad    # Attack/Defense grid search

Elo two-stage search:
  Stage 1: K × home_advantage × goals_scale  (base Elo parameters)
  Stage 2: use_mov × regression_factor × decay_rate  (refinement features)

AD two-stage search:
  Stage 1: k_att × k_def × home_adv_log  (6×6×4 = 144 combos)
  Stage 2: reg × regression_factor       (4×4 = 16 combos)

Prints a ranked table and saves stage-1 results to data/processed/.
"""

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).parent))
from elo_model import mov_factor

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Elo grid definitions
# ---------------------------------------------------------------------------
STAGE1_GRID = {
    "k":               [15, 20, 25, 32, 40, 50],
    "home_advantage":  [50, 75, 100, 125, 150],
    "goals_scale":     [0.10, 0.15, 0.20, 0.25, 0.30],
}
STAGE2_GRID = {
    "use_mov":           [False, True],
    "regression_factor": [0.0, 0.20, 0.30, 0.40],
    "decay_rate":        [0.0, 0.30, 0.50, 0.70],
}

# ---------------------------------------------------------------------------
# AD grid definitions
# ---------------------------------------------------------------------------
AD_STAGE1_GRID = {
    "k_att":         [0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
    "k_def":         [0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
    "home_adv_log":  [0.05, 0.10, 0.15, 0.20],
}
AD_STAGE2_GRID = {
    "reg":               [0.0, 0.001, 0.005, 0.01],
    "regression_factor": [0.0, 0.10,  0.20,  0.30],
}

# ---------------------------------------------------------------------------
# Vectorised Brier scorer
# ---------------------------------------------------------------------------

def _brier_from_probs(ph, pd2, pa, actuals):
    oh = (actuals == "H").astype(float)
    od = (actuals == "D").astype(float)
    oa = (actuals == "A").astype(float)
    return float(np.mean((ph - oh) ** 2 + (pd2 - od) ** 2 + (pa - oa) ** 2))


def _probs_from_ratings(r_h_arr, r_a_arr, home_advantage, avg_home, avg_away, goals_scale):
    """Vectorised Poisson outcome probabilities for Elo model (N matches)."""
    d = (r_h_arr + home_advantage - r_a_arr) / 400.0
    xg_h = avg_home * np.power(10.0,  goals_scale * d)
    xg_a = avg_away * np.power(10.0, -goals_scale * d)

    goals = np.arange(11, dtype=float)

    def pmf_batch(xg):
        log_p = goals * np.log(xg[:, None]) - xg[:, None] - gammaln(goals + 1)
        return np.exp(log_p)

    pmf_h = pmf_batch(xg_h)   # (N, 11)
    pmf_a = pmf_batch(xg_a)   # (N, 11)
    joint  = pmf_h[:, :, None] * pmf_a[:, None, :]  # (N, 11, 11)

    mask_hw = np.tril(np.ones((11, 11), dtype=bool), k=-1)
    mask_d  = np.eye(11, dtype=bool)
    mask_aw = np.triu(np.ones((11, 11), dtype=bool), k=1)

    ph  = joint[:, mask_hw].sum(axis=1)
    pd2 = joint[:, mask_d].sum(axis=1)
    pa  = joint[:, mask_aw].sum(axis=1)
    tot = ph + pd2 + pa
    return ph / tot, pd2 / tot, pa / tot


def _probs_from_ad_ratings(
    atk_h: np.ndarray,
    def_h: np.ndarray,
    atk_a: np.ndarray,
    def_a: np.ndarray,
    home_adv_log: float,
    base_home: float,
    base_away: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised Poisson outcome probabilities for the Attack/Defense model (N matches).

    xg_home = base_home * exp(clip(atk_h - def_a + home_adv_log, -3, 3))
    xg_away = base_away * exp(clip(atk_a - def_h,               -3, 3))
    """
    exp_h = np.clip(atk_h + def_a + home_adv_log, -3.0, 3.0)
    exp_a = np.clip(atk_a + def_h,                -3.0, 3.0)
    xg_h = base_home * np.exp(exp_h)
    xg_a = base_away * np.exp(exp_a)

    goals = np.arange(11, dtype=float)

    def pmf_batch(xg):
        log_p = goals * np.log(xg[:, None]) - xg[:, None] - gammaln(goals + 1)
        return np.exp(log_p)

    pmf_h = pmf_batch(xg_h)   # (N, 11)
    pmf_a = pmf_batch(xg_a)   # (N, 11)
    joint  = pmf_h[:, :, None] * pmf_a[:, None, :]  # (N, 11, 11)

    mask_hw = np.tril(np.ones((11, 11), dtype=bool), k=-1)
    mask_d  = np.eye(11, dtype=bool)
    mask_aw = np.triu(np.ones((11, 11), dtype=bool), k=1)

    ph  = joint[:, mask_hw].sum(axis=1)
    pd2 = joint[:, mask_d].sum(axis=1)
    pa  = joint[:, mask_aw].sum(axis=1)
    tot = ph + pd2 + pa
    return ph / tot, pd2 / tot, pa / tot


# ---------------------------------------------------------------------------
# Elo core backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    matches: pd.DataFrame,
    k: float,
    home_advantage: float,
    goals_scale: float,
    avg_home_goals: float = 1.75,
    avg_away_goals: float = 1.39,
    initial_rating: float = 1500.0,
    use_mov: bool = False,
    regression_factor: float = 0.0,
    decay_rate: float = 0.0,
) -> float:
    """Replay matches, collect pre-match ratings, return Brier score."""
    ref_date = matches["date"].max()

    ratings: dict[str, float] = {}
    current_season = None

    pre_rh, pre_ra, actuals = [], [], []

    for _, row in matches.iterrows():
        home, away   = row["home_team"], row["away_team"]
        hg, ag       = int(row["home_goals"]), int(row["away_goals"])
        season       = row["season"]
        match_date   = row["date"]

        # Season transition → regression to mean
        if season != current_season:
            if current_season is not None and regression_factor > 0:
                for team in list(ratings.keys()):
                    ratings[team] += regression_factor * (initial_rating - ratings[team])
            current_season = season

        r_h = ratings.get(home, initial_rating)
        r_a = ratings.get(away, initial_rating)
        pre_rh.append(r_h)
        pre_ra.append(r_a)
        actuals.append(row["result"])

        # K multiplier (MoV + decay)
        k_mult = 1.0
        if use_mov:
            k_mult *= mov_factor(abs(hg - ag))
        if decay_rate > 0.0:
            days_old = (ref_date - match_date).days
            k_mult *= np.exp(-decay_rate * days_old / 365.0)

        # Elo update
        r_h_adj = r_h + home_advantage
        e_h = 1.0 / (1.0 + 10.0 ** ((r_a - r_h_adj) / 400.0))
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        ratings[home] = r_h + k * k_mult * (s_h - e_h)
        ratings[away] = r_a + k * k_mult * ((1.0 - s_h) - (1.0 - e_h))

    r_h_arr = np.array(pre_rh)
    r_a_arr = np.array(pre_ra)
    actuals_arr = np.array(actuals)

    ph, pd2, pa = _probs_from_ratings(
        r_h_arr, r_a_arr, home_advantage, avg_home_goals, avg_away_goals, goals_scale
    )
    return _brier_from_probs(ph, pd2, pa, actuals_arr)


# ---------------------------------------------------------------------------
# AD core backtest runner
# ---------------------------------------------------------------------------

def run_backtest_ad(
    matches: pd.DataFrame,
    k_att: float,
    k_def: float,
    home_adv_log: float,
    base_home: float = 1.75,
    base_away: float = 1.39,
    reg: float = 0.0,
    regression_factor: float = 0.0,
) -> float:
    """
    Replay all matches with Attack/Defense model, using pre-match ratings
    to compute Poisson probabilities, then return Brier score.

    Same forward-chain discipline as run_backtest: we record ratings BEFORE
    updating so the prediction is always made with pre-match information.
    """
    atk: dict[str, float] = {}
    def_: dict[str, float] = {}
    current_season = None

    pre_atk_h, pre_def_h, pre_atk_a, pre_def_a, actuals = [], [], [], [], []

    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag     = int(row["home_goals"]), int(row["away_goals"])
        season     = row["season"]

        # Season transition → regression toward 0
        if season != current_season:
            if current_season is not None and regression_factor > 0:
                for team in list(atk.keys()):
                    atk[team]  *= (1.0 - regression_factor)
                for team in list(def_.keys()):
                    def_[team] *= (1.0 - regression_factor)
            current_season = season

        # Initialise new teams
        for team in (home, away):
            atk.setdefault(team, 0.0)
            def_.setdefault(team, 0.0)

        # Record PRE-match ratings for Brier computation
        pre_atk_h.append(atk[home])
        pre_def_h.append(def_[home])
        pre_atk_a.append(atk[away])
        pre_def_a.append(def_[away])
        actuals.append(row["result"])

        # Compute xG from pre-match ratings
        exp_h = max(-3.0, min(3.0, atk[home] + def_[away] + home_adv_log))
        exp_a = max(-3.0, min(3.0, atk[away] + def_[home]))
        xg_home = base_home * np.exp(exp_h)
        xg_away = base_away * np.exp(exp_a)

        # Gradient update (def_ += makes it more positive/leaky when conceding more)
        delta_h = hg - xg_home
        delta_a = ag - xg_away

        atk[home]  += k_att * delta_h
        def_[away] += k_def * delta_h
        atk[away]  += k_att * delta_a
        def_[home] += k_def * delta_a

        # Per-match weight decay
        if reg > 0.0:
            for team in {home, away}:
                atk[team]  *= (1.0 - reg)
                def_[team] *= (1.0 - reg)

    atk_h_arr = np.array(pre_atk_h)
    def_h_arr = np.array(pre_def_h)
    atk_a_arr = np.array(pre_atk_a)
    def_a_arr = np.array(pre_def_a)
    actuals_arr = np.array(actuals)

    ph, pd2, pa = _probs_from_ad_ratings(
        atk_h_arr, def_h_arr, atk_a_arr, def_a_arr,
        home_adv_log, base_home, base_away,
    )
    return _brier_from_probs(ph, pd2, pa, actuals_arr)


# ---------------------------------------------------------------------------
# Elo main
# ---------------------------------------------------------------------------

def main_elo() -> dict:
    df = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    n  = len(df)
    print(f"Loaded {n} matches\n")

    # Baseline
    baseline = run_backtest(df, k=32, home_advantage=100, goals_scale=0.20)
    print(f"Baseline (K=32, HA=100, GS=0.20): Brier = {baseline:.5f}\n")

    # ------------------------------------------------------------------ #
    # Stage 1                                                              #
    # ------------------------------------------------------------------ #
    combos1 = list(product(
        STAGE1_GRID["k"],
        STAGE1_GRID["home_advantage"],
        STAGE1_GRID["goals_scale"],
    ))
    print(f"=== Stage 1: {len(combos1)} combos (K x home_adv x goals_scale) ===")

    rows1, best1, best_brier1 = [], {}, float("inf")
    t0 = time.time()

    for i, (k, ha, gs) in enumerate(combos1):
        b = run_backtest(df, k=k, home_advantage=ha, goals_scale=gs)
        rows1.append({"k": k, "home_advantage": ha, "goals_scale": gs, "brier": b})
        if b < best_brier1:
            best_brier1 = b
            best1 = {"k": k, "home_advantage": ha, "goals_scale": gs}
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(combos1)}  best so far: {best_brier1:.5f}  {best1}")

    elapsed = time.time() - t0
    print(f"\nStage 1 done in {elapsed:.1f}s")
    print(f"  Best: {best1}  ->  Brier {best_brier1:.5f}  (vs baseline {baseline:.5f})")

    results1_df = pd.DataFrame(rows1).sort_values("brier")
    results1_df.to_csv(PROCESSED_DIR / "tuning_stage1.csv", index=False)
    print(f"  Top 5:")
    print(results1_df.head(5).to_string(index=False))

    # ------------------------------------------------------------------ #
    # Stage 2                                                              #
    # ------------------------------------------------------------------ #
    combos2 = list(product(
        STAGE2_GRID["use_mov"],
        STAGE2_GRID["regression_factor"],
        STAGE2_GRID["decay_rate"],
    ))
    print(f"\n=== Stage 2: {len(combos2)} combos (MoV x regression x decay) ===")
    print(f"    Fixed: K={best1['k']}, HA={best1['home_advantage']}, GS={best1['goals_scale']}")

    rows2, best2, best_brier2 = [], {}, float("inf")
    t1 = time.time()

    for i, (mov, reg, decay) in enumerate(combos2):
        b = run_backtest(
            df, **best1,
            use_mov=mov, regression_factor=reg, decay_rate=decay,
        )
        rows2.append({"use_mov": mov, "regression_factor": reg, "decay_rate": decay, "brier": b})
        if b < best_brier2:
            best_brier2 = b
            best2 = {"use_mov": mov, "regression_factor": reg, "decay_rate": decay}

    elapsed2 = time.time() - t1
    print(f"Stage 2 done in {elapsed2:.1f}s")
    print(f"  Best: {best2}  ->  Brier {best_brier2:.5f}")

    results2_df = pd.DataFrame(rows2).sort_values("brier")
    print(f"  Top 5:")
    print(results2_df.head(5).to_string(index=False))

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    optimal = {**best1, **best2}
    improvement = (baseline - best_brier2) / baseline * 100

    print(f"\n{'=' * 55}")
    print(f"OPTIMAL ELO PARAMETERS")
    print(f"{'=' * 55}")
    for param, val in optimal.items():
        print(f"  {param:<22} {val}")
    print(f"\n  Brier: {best_brier2:.5f}  (baseline {baseline:.5f}, {improvement:+.1f}%)")
    print(f"{'=' * 55}")

    return optimal


# ---------------------------------------------------------------------------
# AD main
# ---------------------------------------------------------------------------

def main_ad() -> dict:
    df = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    n  = len(df)
    print(f"Loaded {n} matches\n")

    # Naive baseline: always predict base_home / base_away
    baseline = run_backtest_ad(df, k_att=0.0, k_def=0.0, home_adv_log=0.0)
    print(f"AD Baseline (no learning, no home adv): Brier = {baseline:.5f}\n")

    # ------------------------------------------------------------------ #
    # Stage 1: k_att × k_def × home_adv_log                               #
    # ------------------------------------------------------------------ #
    combos1 = list(product(
        AD_STAGE1_GRID["k_att"],
        AD_STAGE1_GRID["k_def"],
        AD_STAGE1_GRID["home_adv_log"],
    ))
    print(f"=== AD Stage 1: {len(combos1)} combos (k_att x k_def x home_adv_log) ===")

    rows1, best1, best_brier1 = [], {}, float("inf")
    t0 = time.time()

    for i, (ka, kd, hal) in enumerate(combos1):
        b = run_backtest_ad(df, k_att=ka, k_def=kd, home_adv_log=hal)
        rows1.append({"k_att": ka, "k_def": kd, "home_adv_log": hal, "brier": b})
        if b < best_brier1:
            best_brier1 = b
            best1 = {"k_att": ka, "k_def": kd, "home_adv_log": hal}
        if (i + 1) % 36 == 0:
            print(f"  {i+1}/{len(combos1)}  best so far: {best_brier1:.5f}  {best1}")

    elapsed = time.time() - t0
    print(f"\nAD Stage 1 done in {elapsed:.1f}s")
    print(f"  Best: {best1}  ->  Brier {best_brier1:.5f}  (vs baseline {baseline:.5f})")

    results1_df = pd.DataFrame(rows1).sort_values("brier")
    results1_df.to_csv(PROCESSED_DIR / "tuning_ad_stage1.csv", index=False)
    print(f"  Top 5:")
    print(results1_df.head(5).to_string(index=False))

    # ------------------------------------------------------------------ #
    # Stage 2: reg × regression_factor                                    #
    # ------------------------------------------------------------------ #
    combos2 = list(product(
        AD_STAGE2_GRID["reg"],
        AD_STAGE2_GRID["regression_factor"],
    ))
    print(f"\n=== AD Stage 2: {len(combos2)} combos (reg x regression_factor) ===")
    print(f"    Fixed: k_att={best1['k_att']}, k_def={best1['k_def']}, "
          f"home_adv_log={best1['home_adv_log']}")

    rows2, best2, best_brier2 = [], {}, float("inf")
    t1 = time.time()

    for i, (reg, rf) in enumerate(combos2):
        b = run_backtest_ad(df, **best1, reg=reg, regression_factor=rf)
        rows2.append({"reg": reg, "regression_factor": rf, "brier": b})
        if b < best_brier2:
            best_brier2 = b
            best2 = {"reg": reg, "regression_factor": rf}

    elapsed2 = time.time() - t1
    print(f"AD Stage 2 done in {elapsed2:.1f}s")
    print(f"  Best: {best2}  ->  Brier {best_brier2:.5f}")

    results2_df = pd.DataFrame(rows2).sort_values("brier")
    print(f"  Top 5:")
    print(results2_df.head(5).to_string(index=False))

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    optimal = {**best1, **best2}
    improvement = (baseline - best_brier2) / baseline * 100

    print(f"\n{'=' * 55}")
    print(f"OPTIMAL AD PARAMETERS")
    print(f"{'=' * 55}")
    for param, val in optimal.items():
        print(f"  {param:<22} {val}")
    print(f"\n  Brier: {best_brier2:.5f}  (baseline {baseline:.5f}, {improvement:+.1f}%)")
    print(f"  Copy these into model_config.py -> AD_PARAMS")
    print(f"{'=' * 55}")

    return optimal


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune model parameters via grid search")
    parser.add_argument(
        "--model",
        choices=["elo", "ad"],
        default="elo",
        help="Which model to tune: 'elo' (default) or 'ad' (attack/defense)",
    )
    args = parser.parse_args()

    if args.model == "ad":
        main_ad()
    else:
        main_elo()
