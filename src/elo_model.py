"""
elo_model.py — Phase 3, Step 8

Implements an Elo-based rating system for soccer teams.

Core formulas
-------------
Expected score (win probability, ignoring draws):
    E = 1 / (1 + 10 ^ ((opponent_rating - own_rating) / 400))

Rating update after a match:
    new_rating = old_rating + K * k_multiplier * (actual_score - expected_score)
    where actual_score = 1 (win), 0.5 (draw), 0 (loss)
    and k_multiplier combines margin-of-victory and time-decay adjustments.

Margin-of-victory multiplier (optional):
    mov_factor = min((goal_diff + 2) / 3, 2.0)
    1-goal margin → 1.0x, 2-goal → 1.33x, 3-goal → 1.67x, 4+ → 2.0x (capped)

Time-decay multiplier (optional):
    decay_factor = exp(-decay_rate * days_since_match / 365)
    Recent matches get full K; matches from 2 years ago get K * ~0.37 (rate=0.5).

Expected goals conversion:
    xg_home = avg_home_goals * 10 ^ ( goals_scale * d)
    xg_away = avg_away_goals * 10 ^ (-goals_scale * d)
    where d = (home_adj_rating - away_rating) / 400

Win / draw / loss probabilities:
    Computed by summing exact Poisson probabilities over a grid of scorelines.
    P(h:a) = Poisson(h | xg_home) * Poisson(a | xg_away)

Tunable parameters:
    k              = 32      learning rate          range: 15–50
    home_advantage = 100     added to home rating   range: 50–150
    goals_scale    = 0.20    rating diff -> xG      range: 0.10–0.30
    use_mov        = False   margin-of-victory K scaling
    regression_factor = 0.0 season-start pull toward mean (0=none, 0.3=30%)
    decay_rate     = 0.0     time-decay on K (0=none, 0.5=half-life ~1.4yr)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from scipy.stats import poisson

# Maximum scoreline considered when computing outcome probabilities.
_MAX_GOALS = 10


def mov_factor(goal_diff: int) -> float:
    """
    Margin-of-victory multiplier for the K-factor.
    Rewards larger wins with stronger rating updates.

    1-goal: 1.00x  |  2-goal: 1.33x  |  3-goal: 1.67x  |  4+: 2.00x (cap)
    Formula: min((|goal_diff| + 2) / 3, 2.0)
    """
    return min((abs(goal_diff) + 2) / 3.0, 2.0)


@dataclass
class EloRating:
    """
    Maintains Elo ratings for a set of teams and generates match predictions.

    Parameters
    ----------
    k : float
        Base learning rate.
    home_advantage : float
        Rating points added to the home team's effective rating.
    initial_rating : float
        Starting rating for any team not yet seen.
    avg_home_goals : float
        League-average home goals per match (xG baseline).
    avg_away_goals : float
        League-average away goals per match (xG baseline).
    goals_scale : float
        How steeply rating difference maps to goal expectation.
    use_mov : bool
        If True, scale K by margin-of-victory multiplier.
    regression_factor : float
        Fraction to pull ratings toward initial_rating at each season start.
        0.0 = no regression, 0.3 = pull 30% toward mean.
    decay_rate : float
        Exponential decay rate applied to K for older matches.
        0.0 = no decay; 0.5 means a match 1.4 years old gets ~0.5x K.
    """

    k: float = 32.0
    home_advantage: float = 100.0
    initial_rating: float = 1500.0
    avg_home_goals: float = 1.75
    avg_away_goals: float = 1.39
    goals_scale: float = 0.20
    use_mov: bool = False
    regression_factor: float = 0.0
    decay_rate: float = 0.0

    ratings: dict[str, float] = field(default_factory=dict)
    _current_season: str | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Basic accessors                                                      #
    # ------------------------------------------------------------------ #

    def get_rating(self, team: str) -> float:
        """Return current rating; new teams start at initial_rating."""
        return self.ratings.get(team, self.initial_rating)

    def all_ratings(self) -> dict[str, float]:
        """Return a copy of the current ratings dict, sorted by rating."""
        return dict(sorted(self.ratings.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------ #
    # Core Elo math                                                        #
    # ------------------------------------------------------------------ #

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Probability that team A beats team B in a head-to-head contest
        (draw-inclusive — this is the standard Elo expected score).

        E_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def apply_season_regression(self) -> None:
        """
        Pull all current ratings toward initial_rating by regression_factor.
        Call this at the start of each new season (before processing its matches).

        Example: regression_factor=0.30, team at 1700
            new_rating = 1700 + 0.30 * (1500 - 1700) = 1640
        """
        if self.regression_factor <= 0.0:
            return
        for team in list(self.ratings.keys()):
            self.ratings[team] += self.regression_factor * (
                self.initial_rating - self.ratings[team]
            )

    def update_rating(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        season: str | None = None,
        match_date: date | None = None,
        reference_date: date | None = None,
    ) -> tuple[float, float]:
        """
        Update ratings after a match and return the new (home, away) ratings.

        Applies season regression automatically when the season changes.
        Scales K by margin-of-victory and/or time-decay if enabled.

        Parameters
        ----------
        home_team, away_team : str
        home_goals, away_goals : int
        season : str, optional
            Season code (e.g. '2526'). Triggers regression on season change.
        match_date : date, optional
            Date of the match. Required for time-decay.
        reference_date : date, optional
            The most recent match date (used as "today" for decay calculation).

        Returns
        -------
        (new_home_rating, new_away_rating)
        """
        # Season change → apply regression before processing first match
        if season is not None and season != self._current_season:
            if self._current_season is not None:
                self.apply_season_regression()
            self._current_season = season

        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)

        r_home_adj = r_home + self.home_advantage
        e_home = self._expected_score(r_home_adj, r_away)
        e_away = 1.0 - e_home

        if home_goals > away_goals:
            s_home, s_away = 1.0, 0.0
        elif home_goals == away_goals:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0

        # Compose K multiplier
        k_mult = 1.0
        if self.use_mov:
            k_mult *= mov_factor(abs(home_goals - away_goals))
        if self.decay_rate > 0.0 and match_date is not None and reference_date is not None:
            days_old = (reference_date - match_date).days
            k_mult *= math.exp(-self.decay_rate * days_old / 365.0)

        effective_k = self.k * k_mult
        new_home = r_home + effective_k * (s_home - e_home)
        new_away = r_away + effective_k * (s_away - e_away)

        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away

        return new_home, new_away

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float]:
        """
        Estimate expected goals for a fixture using the teams' current ratings.

        The rating difference (with home advantage applied) is used to scale
        league-average goals up or down:

            d = (home_rating + home_advantage - away_rating) / 400
            xg_home = avg_home_goals * 10 ^ ( goals_scale * d)
            xg_away = avg_away_goals * 10 ^ (-goals_scale * d)

        Example (goals_scale=0.20, home_advantage=100):
            Even match  (both 1500) → d = 100/400 = 0.25
              xg_home = 1.75 * 10^0.050 ≈ 1.97
              xg_away = 1.39 * 10^-0.050 ≈ 1.24

            Strong home favourite (+300 net rating diff):
              d = 400/400 = 1.0
              xg_home = 1.75 * 10^0.20 ≈ 2.77
              xg_away = 1.39 * 10^-0.20 ≈ 0.88

        Returns
        -------
        (xg_home, xg_away)
        """
        r_home = self.get_rating(home_team) + self.home_advantage
        r_away = self.get_rating(away_team)
        d = (r_home - r_away) / 400.0

        xg_home = self.avg_home_goals * (10.0 ** (self.goals_scale * d))
        xg_away = self.avg_away_goals * (10.0 ** (-self.goals_scale * d))
        return xg_home, xg_away

    def predict_outcome(
        self,
        home_team: str,
        away_team: str,
    ) -> dict:
        """
        Predict match outcome probabilities and expected goals.

        Sums exact Poisson probabilities over all scorelines up to
        _MAX_GOALS x _MAX_GOALS.

        Returns
        -------
        dict with keys:
            home_win_prob  : float [0, 1]
            draw_prob      : float [0, 1]
            away_win_prob  : float [0, 1]
            xg_home        : float
            xg_away        : float
            home_rating    : float (before this match)
            away_rating    : float (before this match)
        """
        xg_home, xg_away = self.expected_goals(home_team, away_team)

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

        # Normalise to account for truncation at _MAX_GOALS
        total = p_home_win + p_draw + p_away_win
        p_home_win /= total
        p_draw /= total
        p_away_win /= total

        return {
            "home_win_prob": round(p_home_win, 4),
            "draw_prob": round(p_draw, 4),
            "away_win_prob": round(p_away_win, 4),
            "xg_home": round(xg_home, 2),
            "xg_away": round(xg_away, 2),
            "home_rating": round(self.get_rating(home_team), 1),
            "away_rating": round(self.get_rating(away_team), 1),
        }


# ------------------------------------------------------------------ #
# Quick smoke test                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    elo = EloRating()

    # Feed in a few results
    sample = [
        ("Bayern Munich", "Schalke 04", 8, 0),
        ("Dortmund", "Bayern Munich", 2, 3),
        ("Leverkusen", "Bayern Munich", 1, 0),
    ]
    for home, away, hg, ag in sample:
        elo.update_rating(home, away, hg, ag)

    print("Current ratings:")
    for team, rating in elo.all_ratings().items():
        print(f"  {team:20s}  {rating:.1f}")

    print("\nPrediction: Bayern Munich vs Dortmund")
    pred = elo.predict_outcome("Bayern Munich", "Dortmund")
    for k, v in pred.items():
        print(f"  {k}: {v}")
