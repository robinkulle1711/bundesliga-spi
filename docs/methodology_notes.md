# Methodology Notes: Bundesliga SPI Model

Reference: [FiveThirtyEight SPI Methodology](https://fivethirtyeight.com/methodology/how-our-club-soccer-predictions-work/)

---

## Core Components

### 1. SPI Ratings (Offensive / Defensive Strength)

FiveThirtyEight's SPI has two components per team:
- **Offensive rating** — expected goals scored per match vs. average opponent
- **Defensive rating** — expected goals conceded per match vs. average opponent
- **Overall SPI** — combined measure of team quality (0–100 scale)

Our simpler Elo implementation uses a single rating per team (initially 1500).

---

### 2. Elo Rating Update

**Expected score (probability of winning):**
```
E = 1 / (1 + 10^((opponent_rating - own_rating) / 400))
```

**Rating update after match:**
```
new_rating = old_rating + K * (actual_score - expected_score)
```

Where:
- `actual_score` = 1 (win), 0.5 (draw), 0 (loss)
- `K` = learning rate (start at 32, tune later — range 20–40)

---

### 3. Home Field Advantage

Add a fixed offset to the home team's effective rating before computing `E`:
```
home_effective_rating = home_rating + HOME_ADVANTAGE
```
Start with `HOME_ADVANTAGE = 100` (tune to data).

---

### 4. Win / Draw / Loss Probabilities

From expected score `E` (home team's win probability):
- `P(home win)` ≈ E (adjusted for draw zone)
- `P(away win)` ≈ 1 - E (adjusted for draw zone)
- `P(draw)` — derived from a draw probability model or fixed parameter

FiveThirtyEight uses a more complex mapping from goal expectation to outcome probabilities via the Poisson distribution. We will implement this in Phase 3.

---

### 5. Expected Goals (xG)

FiveThirtyEight estimates expected goals from team ratings:
```
xG_home = f(home_offensive_rating, away_defensive_rating)
xG_away = f(away_offensive_rating, home_defensive_rating)
```

With single Elo ratings, a simple proxy:
```
xG_home ≈ league_avg_goals * (home_rating / avg_rating)
xG_away ≈ league_avg_goals * (away_rating / avg_rating)
```

---

## Parameters to Tune (Phase 3)

| Parameter       | Initial Value | Range to Try |
|-----------------|---------------|--------------|
| K-factor        | 32            | 20 – 40      |
| Home advantage  | 100           | 50 – 150     |
| Initial rating  | 1500          | fixed        |
| Draw zone width | TBD           | TBD          |

---

## Validation Metrics (Phase 3)

- **Accuracy** — % of correct outcome predictions (win/draw/loss)
- **Brier score** — mean squared error of probability predictions (lower = better)
- **Log loss** — penalizes confident wrong predictions (lower = better)
- **Calibration** — are 70% predictions correct ~70% of the time?
- **Baseline comparison** — vs. always predicting home win, vs. league position

---

## Key Differences from FiveThirtyEight

| FiveThirtyEight | Our Model |
|-----------------|-----------|
| Separate off/def ratings | Single Elo rating |
| xG-based updates | Score-based updates |
| Cross-league calibration | Bundesliga only |
| Poisson goal model | Simplified proxy |
| Regression to mean | Not initially |

These simplifications are intentional for Phase 1. Enhancements are planned for later phases.
