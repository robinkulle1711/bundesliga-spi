"""
attack_defense_model.py — Poisson Attack/Defense Rating System

Replaces the single Elo number with separate attack and defense parameters
per team. This enables:
  - Goal-level predictions (xG per team) with independent attack/defense signals
  - Richer scatter plot: attack vs defensive_solidity
  - Better Brier scores because large wins (5-0) produce larger deltas

Sign convention
---------------
  def_[team] > 0  → weaker defense (concedes more than expected)
  def_[team] < 0  → stronger defense (concedes less than expected)
  defensive_solidity = -def_  → higher = better defense (for display)
  power = atk - def_          → subtracts defensive "leakiness"

Core formulas
-------------
Expected goals:
    xg_home = base_home * exp(clip(atk[home] + def_[away] + home_adv_log, -3, 3))
    xg_away = base_away * exp(clip(atk[away] + def_[home],               -3, 3))

  When def_[away] is large (leaky defense) → more xG home  ✓
  When def_[away] is negative (solid defense) → fewer xG home  ✓

Update rule (Poisson gradient ascent, MoV is implicit via delta size):
    delta_h = home_goals - xg_home
    atk[home]  += k_att * delta_h     # home attack rises if scored more than expected
    def_[away] += k_def * delta_h     # away def_ rises if conceded more than expected (weaker)

    delta_a = away_goals - xg_away
    atk[away]  += k_att * delta_a
    def_[home] += k_def * delta_a

    # Optional per-match weight decay:
    atk[team] *= (1 - reg);  def_[team] *= (1 - reg)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from scipy.stats import poisson

_MAX_GOALS = 10


@dataclass
class AttackDefenseRating:
    """
    Maintains separate attack and defense ratings for a set of teams.

    Parameters
    ----------
    k_att : float
        Learning rate for attack parameter updates.
    k_def : float
        Learning rate for defense parameter updates.
    home_adv_log : float
        Home advantage in log-goal space (added to home team's attack exponent).
    base_home : float
        League-average home goals per match (baseline xG).
    base_away : float
        League-average away goals per match (baseline xG).
    reg : float
        Per-match weight decay applied to atk/def_ (0 = none).
    regression_factor : float
        Fraction to pull atk/def_ toward 0 at each season start (0 = none).
    """

    k_att: float = 0.08
    k_def: float = 0.08
    home_adv_log: float = 0.10
    base_home: float = 1.75
    base_away: float = 1.39
    reg: float = 0.0
    regression_factor: float = 0.0

    atk: dict[str, float] = field(default_factory=dict)
    def_: dict[str, float] = field(default_factory=dict)
    _current_season: str | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    def get_attack(self, team: str) -> float:
        """Return current attack rating (new teams start at 0.0)."""
        return self.atk.get(team, 0.0)

    def get_defense(self, team: str) -> float:
        """
        Return raw defense parameter.
        Positive = weaker defense (concedes more than expected).
        Negative = stronger defense (concedes less than expected).
        For display, use defensive_solidity = -get_defense() so higher = better.
        """
        return self.def_.get(team, 0.0)

    def get_power(self, team: str) -> float:
        """
        Composite power = atk - def_ (higher = stronger overall).
        Subtracts def_ because positive def_ = leaky defense = worse team.
        """
        return self.get_attack(team) - self.get_defense(team)

    def all_ratings(self) -> list[dict]:
        """
        Return list of {team, attack, defense, power} dicts sorted by power desc.
        'defense' column is raw def_ (use -defense for defensive_solidity display).
        """
        all_teams = set(self.atk) | set(self.def_)
        rows = []
        for team in all_teams:
            a = self.get_attack(team)
            d = self.get_defense(team)
            rows.append({
                "team":    team,
                "attack":  round(a, 4),
                "defense": round(d, 4),
                "power":   round(a - d, 4),
            })
        return sorted(rows, key=lambda r: r["power"], reverse=True)

    # ------------------------------------------------------------------ #
    # Core math                                                            #
    # ------------------------------------------------------------------ #

    def _expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """
        Compute expected goals for a fixture using current ratings.

        xg_home = base_home * exp(clip(atk[home] + def_[away] + home_adv_log, -3, 3))
        xg_away = base_away * exp(clip(atk[away] + def_[home],               -3, 3))

        def_[away] > 0 → leaky defense → more xG home
        def_[away] < 0 → solid defense → fewer xG home
        """
        exponent_h = self.get_attack(home) + self.get_defense(away) + self.home_adv_log
        exponent_a = self.get_attack(away) + self.get_defense(home)
        xg_home = self.base_home * math.exp(max(-3.0, min(3.0, exponent_h)))
        xg_away = self.base_away * math.exp(max(-3.0, min(3.0, exponent_a)))
        return xg_home, xg_away

    def apply_season_regression(self) -> None:
        """
        Pull all atk/def_ toward 0 by regression_factor at season start.
        Example: regression_factor=0.30, atk=0.5 → new_atk = 0.5 * (1 - 0.30) = 0.35
        """
        if self.regression_factor <= 0.0:
            return
        for team in list(self.atk.keys()):
            self.atk[team]  *= (1.0 - self.regression_factor)
        for team in list(self.def_.keys()):
            self.def_[team] *= (1.0 - self.regression_factor)

    def update_rating(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        season: str | None = None,
        match_date: date | None = None,
        reference_date: date | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Update attack/defense ratings after a match.

        Applies season regression automatically when the season changes.
        MoV is implicit: 5-0 produces a larger delta than 1-0.

        Parameters
        ----------
        home_team, away_team : str
        home_goals, away_goals : int
        season : str, optional
            Season code (e.g. '2526'). Triggers regression on season change.
        match_date, reference_date : date, optional
            Unused by this model but kept for API compatibility with EloRating.

        Returns
        -------
        (atk_home_post, def_home_post, atk_away_post, def_away_post)
        """
        # Season change → apply regression before processing first match
        if season is not None and season != self._current_season:
            if self._current_season is not None:
                self.apply_season_regression()
            self._current_season = season

        # Initialise new teams at 0
        for team in (home_team, away_team):
            self.atk.setdefault(team, 0.0)
            self.def_.setdefault(team, 0.0)

        xg_home, xg_away = self._expected_goals(home_team, away_team)

        delta_h = home_goals - xg_home
        delta_a = away_goals - xg_away

        # Home team scored delta_h more than expected: home attack up, away def_ up (leakier)
        self.atk[home_team] += self.k_att * delta_h
        self.def_[away_team] += self.k_def * delta_h

        # Away team scored delta_a more than expected: away attack up, home def_ up (leakier)
        self.atk[away_team] += self.k_att * delta_a
        self.def_[home_team] += self.k_def * delta_a

        # Per-match weight decay
        if self.reg > 0.0:
            for team in {home_team, away_team}:
                self.atk[team]  *= (1.0 - self.reg)
                self.def_[team] *= (1.0 - self.reg)

        return (
            self.atk[home_team],
            self.def_[home_team],
            self.atk[away_team],
            self.def_[away_team],
        )

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict_outcome(self, home_team: str, away_team: str) -> dict:
        """
        Predict match outcome probabilities and expected goals.

        Sums exact Poisson probabilities over all scorelines up to
        _MAX_GOALS x _MAX_GOALS.

        Returns
        -------
        dict with keys:
            home_win_prob, draw_prob, away_win_prob : float [0, 1]
            xg_home, xg_away                        : float
            home_attack, home_defense               : float (raw ratings)
            away_attack, away_defense               : float (raw ratings)
            home_power, away_power                  : float (atk - def_)
        """
        xg_home, xg_away = self._expected_goals(home_team, away_team)

        p_home_win = 0.0
        p_draw = 0.0
        p_away_win = 0.0

        for h in range(_MAX_GOALS + 1):
            ph = poisson.pmf(h, xg_home)
            for a in range(_MAX_GOALS + 1):
                pa = poisson.pmf(a, xg_away)
                p = ph * pa
                if h > a:
                    p_home_win += p
                elif h == a:
                    p_draw += p
                else:
                    p_away_win += p

        total = p_home_win + p_draw + p_away_win
        p_home_win /= total
        p_draw /= total
        p_away_win /= total

        return {
            "home_win_prob":  round(p_home_win, 4),
            "draw_prob":      round(p_draw, 4),
            "away_win_prob":  round(p_away_win, 4),
            "xg_home":        round(xg_home, 2),
            "xg_away":        round(xg_away, 2),
            "home_attack":    round(self.get_attack(home_team), 4),
            "home_defense":   round(self.get_defense(home_team), 4),
            "away_attack":    round(self.get_attack(away_team), 4),
            "away_defense":   round(self.get_defense(away_team), 4),
            "home_power":     round(self.get_power(home_team), 4),
            "away_power":     round(self.get_power(away_team), 4),
        }


# ------------------------------------------------------------------ #
# Smoke test                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    model = AttackDefenseRating()

    sample = [
        ("Bayern Munich", "Schalke 04", 8, 0),
        ("Dortmund",      "Bayern Munich", 2, 3),
        ("Leverkusen",    "Bayern Munich", 1, 0),
    ]
    for home, away, hg, ag in sample:
        model.update_rating(home, away, hg, ag)

    print("Current ratings (power desc):")
    for r in model.all_ratings():
        ds = -r["defense"]
        print(
            f"  {r['team']:<20}  "
            f"atk={r['attack']:+.3f}  "
            f"def_={r['defense']:+.3f}  "
            f"def_sol={ds:+.3f}  "
            f"power={r['power']:+.3f}"
        )

    print("\nSanity check after Bayern 8-0 Schalke:")
    print("  Bayern  attack should be   POSITIVE (scored 8)          :", f"{model.get_attack('Bayern Munich'):+.4f}")
    print("  Schalke def_ should be     POSITIVE (conceded 8, leaky) :", f"{model.get_defense('Schalke 04'):+.4f}")
    print("  Schalke attack should be   NEGATIVE (scored 0)          :", f"{model.get_attack('Schalke 04'):+.4f}")
    print("  Bayern  def_ should be     NEGATIVE (conceded 0, solid) :", f"{model.get_defense('Bayern Munich'):+.4f}")
    print()
    print("  Schalke defensive_solidity (-def_) should be NEGATIVE   :", f"{-model.get_defense('Schalke 04'):+.4f}")
    print("  Bayern  defensive_solidity (-def_) should be POSITIVE   :", f"{-model.get_defense('Bayern Munich'):+.4f}")
    print()
    print("  Bayern power should be GREATER than Schalke:")
    print(f"    Bayern  power = {model.get_power('Bayern Munich'):+.4f}")
    print(f"    Schalke power = {model.get_power('Schalke 04'):+.4f}")

    print("\nPrediction: Bayern Munich vs Dortmund")
    pred = model.predict_outcome("Bayern Munich", "Dortmund")
    for k, v in pred.items():
        print(f"  {k}: {v}")
