"""
model_config.py — single source of truth for model parameters.

Import ELO_PARAMS or AD_PARAMS from here instead of duplicating
the dicts across calculate_ratings.py, predict_matches.py, and dashboard.py.
"""

# ---------------------------------------------------------------------------
# Elo model — kept for historical Brier reference (0.60814)
# ---------------------------------------------------------------------------
ELO_PARAMS = dict(
    k=40,
    home_advantage=50,
    initial_rating=1500,
    avg_home_goals=1.75,
    avg_away_goals=1.39,
    goals_scale=0.25,
    use_mov=True,
    regression_factor=0.0,
    decay_rate=0.0,
)

# ---------------------------------------------------------------------------
# Attack / Defense model
# Starting values — run tune_params.py --model ad to optimise.
# ---------------------------------------------------------------------------
AD_PARAMS = dict(
    k_att=0.02,
    k_def=0.02,
    home_adv_log=0.05,
    base_home=1.75,
    base_away=1.39,
    reg=0.0,
    regression_factor=0.0,
)
# Tuning result: Brier 0.60022 (baseline 0.64877, +7.5% improvement)
# Beats optimised Elo: 0.60814 → 0.60022
