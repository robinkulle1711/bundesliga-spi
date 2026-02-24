"""
dashboard.py — Bundesliga SPI Dashboard

Streamlit dashboard with a sidebar model toggle: Attack/Defense or Elo.
All 4 tabs respond to the selection.

Run with: python -m streamlit run src/dashboard.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.special import gammaln

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from model_config import AD_PARAMS, ELO_PARAMS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bundesliga Prediction Model",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_current_ratings() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / "data/processed/current_ratings.csv")


@st.cache_data
def load_history() -> pd.DataFrame:
    df = pd.read_csv(
        BASE_DIR / "data/processed/team_ratings_history.csv",
        parse_dates=["date"],
    )
    # Validate expected columns exist — catches stale cache from old CSV schema
    required = ["home_elo_pre", "away_elo_pre", "home_elo_post", "away_elo_post"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.cache_data.clear()
        df = pd.read_csv(
            BASE_DIR / "data/processed/team_ratings_history.csv",
            parse_dates=["date"],
        )
    return df


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = BASE_DIR / "data/predictions/upcoming_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def build_ad_timeseries(history_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format: date, team, metric, value  (metric = attack/defensive_solidity/power)."""
    rows = []
    for side, team_col, atk_col, def_col in [
        ("home", "home_team", "home_atk_post", "home_def_post"),
        ("away", "away_team", "away_atk_post", "away_def_post"),
    ]:
        sub = history_df[["date", team_col, atk_col, def_col]].copy()
        sub.columns = ["date", "team", "atk", "def_"]
        atk_rows = sub[["date", "team", "atk"]].copy()
        atk_rows.columns = ["date", "team", "value"]
        atk_rows["metric"] = "attack"

        ds_rows = sub[["date", "team", "def_"]].copy()
        ds_rows.columns = ["date", "team", "value"]
        ds_rows["value"] = -ds_rows["value"]
        ds_rows["metric"] = "defensive_solidity"

        pwr_rows = sub[["date", "team"]].copy()
        pwr_rows["value"] = sub["atk"] - sub["def_"]
        pwr_rows["metric"] = "power"

        rows.extend([atk_rows, ds_rows, pwr_rows])

    return pd.concat(rows).sort_values(["team", "metric", "date"]).reset_index(drop=True)


@st.cache_data
def build_elo_timeseries(history_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format: date, team, value  (elo rating after each match)."""
    home = history_df[["date", "home_team", "home_elo_post"]].rename(
        columns={"home_team": "team", "home_elo_post": "value"}
    )
    away = history_df[["date", "away_team", "away_elo_post"]].rename(
        columns={"away_team": "team", "away_elo_post": "value"}
    )
    return pd.concat([home, away]).sort_values(["team", "date"]).reset_index(drop=True)


@st.cache_data
def compute_ad_backtest_metrics(history_df: pd.DataFrame) -> dict:
    """Backtest using AD pre-match ratings."""
    home_adv_log = AD_PARAMS["home_adv_log"]
    base_home    = AD_PARAMS["base_home"]
    base_away    = AD_PARAMS["base_away"]

    atk_h = history_df["home_atk_pre"].values
    def_h = history_df["home_def_pre"].values
    atk_a = history_df["away_atk_pre"].values
    def_a = history_df["away_def_pre"].values
    results = history_df["result"].values

    exp_h = np.clip(atk_h + def_a + home_adv_log, -3.0, 3.0)
    exp_a = np.clip(atk_a + def_h,                -3.0, 3.0)
    xg_h  = base_home * np.exp(exp_h)
    xg_a  = base_away * np.exp(exp_a)

    ph, pd_, pa = _poisson_probs(xg_h, xg_a)
    oh = (results == "H").astype(float)
    od = (results == "D").astype(float)
    oa = (results == "A").astype(float)

    brier          = float(np.mean((ph - oh)**2 + (pd_ - od)**2 + (pa - oa)**2))
    brier_baseline = float(np.mean((1 - oh)**2 + od**2 + oa**2))
    accuracy       = float(np.mean(_pred_label(ph, pd_, pa) == results))
    baseline_acc   = float(np.mean(results == "H"))

    actual_hg = history_df["home_goals"].values.astype(float)
    actual_ag = history_df["away_goals"].values.astype(float)
    rmse_home       = float(np.sqrt(np.mean((xg_h - actual_hg)**2)))
    rmse_away       = float(np.sqrt(np.mean((xg_a - actual_ag)**2)))
    rmse_naive_home = float(np.sqrt(np.mean((base_home - actual_hg)**2)))
    rmse_naive_away = float(np.sqrt(np.mean((base_away - actual_ag)**2)))

    return {
        "accuracy": accuracy, "baseline_accuracy": baseline_acc,
        "brier": brier, "brier_baseline": brier_baseline,
        "n_matches": len(results),
        "rmse_home": rmse_home, "rmse_away": rmse_away,
        "rmse_naive_home": rmse_naive_home, "rmse_naive_away": rmse_naive_away,
        "xg_h": xg_h, "actual_hg": actual_hg,
        "xg_a": xg_a, "actual_ag": actual_ag,
        "calibration": _calibration(ph, oh),
    }


@st.cache_data
def compute_elo_backtest_metrics(history_df: pd.DataFrame) -> dict:
    """Backtest using Elo pre-match ratings."""
    HOME_ADV    = ELO_PARAMS["home_advantage"]
    AVG_HOME    = ELO_PARAMS["avg_home_goals"]
    AVG_AWAY    = ELO_PARAMS["avg_away_goals"]
    GOALS_SCALE = ELO_PARAMS["goals_scale"]

    r_h     = history_df["home_elo_pre"].values
    r_a     = history_df["away_elo_pre"].values
    results = history_df["result"].values

    d    = (r_h + HOME_ADV - r_a) / 400.0
    xg_h = AVG_HOME * np.power(10.0,  GOALS_SCALE * d)
    xg_a = AVG_AWAY * np.power(10.0, -GOALS_SCALE * d)

    ph, pd_, pa = _poisson_probs(xg_h, xg_a)
    oh = (results == "H").astype(float)
    od = (results == "D").astype(float)
    oa = (results == "A").astype(float)

    brier          = float(np.mean((ph - oh)**2 + (pd_ - od)**2 + (pa - oa)**2))
    brier_baseline = float(np.mean((1 - oh)**2 + od**2 + oa**2))
    accuracy       = float(np.mean(_pred_label(ph, pd_, pa) == results))
    baseline_acc   = float(np.mean(results == "H"))

    return {
        "accuracy": accuracy, "baseline_accuracy": baseline_acc,
        "brier": brier, "brier_baseline": brier_baseline,
        "n_matches": len(results),
        "calibration": _calibration(ph, oh),
    }


@st.cache_data
def season_accuracy_ad(history_df: pd.DataFrame) -> pd.DataFrame:
    home_adv_log = AD_PARAMS["home_adv_log"]
    base_home    = AD_PARAMS["base_home"]
    base_away    = AD_PARAMS["base_away"]
    rows = []
    for season, grp in history_df.groupby("season"):
        exp_h = np.clip(grp["home_atk_pre"].values + grp["away_def_pre"].values + home_adv_log, -3, 3)
        exp_a = np.clip(grp["away_atk_pre"].values + grp["home_def_pre"].values,                -3, 3)
        xg_h  = base_home * np.exp(exp_h)
        xg_a  = base_away * np.exp(exp_a)
        ph, pd_, pa = _poisson_probs(xg_h, xg_a)
        actual = grp["result"].values
        rows.append({
            "season": season,
            "accuracy": float(np.mean(_pred_label(ph, pd_, pa) == actual)),
            "baseline": float(np.mean(actual == "H")),
            "matches": len(grp),
        })
    return pd.DataFrame(rows)


@st.cache_data
def season_accuracy_elo(history_df: pd.DataFrame) -> pd.DataFrame:
    HOME_ADV    = ELO_PARAMS["home_advantage"]
    AVG_HOME    = ELO_PARAMS["avg_home_goals"]
    AVG_AWAY    = ELO_PARAMS["avg_away_goals"]
    GOALS_SCALE = ELO_PARAMS["goals_scale"]
    rows = []
    for season, grp in history_df.groupby("season"):
        d    = (grp["home_elo_pre"].values + HOME_ADV - grp["away_elo_pre"].values) / 400.0
        xg_h = AVG_HOME * np.power(10.0,  GOALS_SCALE * d)
        xg_a = AVG_AWAY * np.power(10.0, -GOALS_SCALE * d)
        ph, pd_, pa = _poisson_probs(xg_h, xg_a)
        actual = grp["result"].values
        rows.append({
            "season": season,
            "accuracy": float(np.mean(_pred_label(ph, pd_, pa) == actual)),
            "baseline": float(np.mean(actual == "H")),
            "matches": len(grp),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared math helpers
# ---------------------------------------------------------------------------

def _poisson_probs(xg_h: np.ndarray, xg_a: np.ndarray):
    goals = np.arange(11, dtype=float)
    def pmf(xg):
        log_p = goals * np.log(xg[:, None]) - xg[:, None] - gammaln(goals + 1)
        return np.exp(log_p)
    pmf_h = pmf(xg_h)
    pmf_a = pmf(xg_a)
    joint = pmf_h[:, :, None] * pmf_a[:, None, :]
    mask_hw = np.tril(np.ones((11, 11), dtype=bool), k=-1)
    mask_d  = np.eye(11, dtype=bool)
    mask_aw = np.triu(np.ones((11, 11), dtype=bool), k=1)
    ph  = joint[:, mask_hw].sum(axis=1)
    pd_ = joint[:, mask_d].sum(axis=1)
    pa  = joint[:, mask_aw].sum(axis=1)
    tot = ph + pd_ + pa
    return ph / tot, pd_ / tot, pa / tot


def _pred_label(ph, pd_, pa):
    return np.where((ph >= pd_) & (ph >= pa), "H", np.where(pd_ >= pa, "D", "A"))


def _calibration(ph: np.ndarray, oh: np.ndarray) -> pd.DataFrame:
    bins = np.arange(0, 1.1, 0.1)
    labels, rates, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (ph >= lo) & (ph < hi)
        if mask.sum() > 0:
            labels.append(f"{lo:.0%}–{hi:.0%}")
            rates.append(oh[mask].mean())
            counts.append(mask.sum())
    return pd.DataFrame({"predicted_range": labels, "actual_rate": rates, "count": counts})


def _season_label(s) -> str:
    s = str(s)
    return f"20{s[:2]}-{s[2:]}"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ratings_df = load_current_ratings()
history_df = load_history()
pred_df    = load_predictions()
ad_ts_df   = build_ad_timeseries(history_df)
elo_ts_df  = build_elo_timeseries(history_df)

current_teams = (
    set(history_df[history_df["season"] == history_df["season"].max()]["home_team"])
    | set(history_df[history_df["season"] == history_df["season"].max()]["away_team"])
)
current_ratings = (
    ratings_df[ratings_df["team"].isin(current_teams)]
    .sort_values("power", ascending=False)
    .reset_index(drop=True)
)
current_ratings.index = range(1, len(current_ratings) + 1)
current_ratings["defensive_solidity"] = -current_ratings["defense"]

# Elo-sorted view for Tab 1
current_elo = (
    current_ratings.sort_values("elo_rating", ascending=False)
    .reset_index(drop=True)
)
current_elo.index = range(1, len(current_elo) + 1)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚽ Bundesliga SPI")
    st.caption("FiveThirtyEight-style prediction model")
    st.divider()

    model_choice = st.radio(
        "Model",
        ["⚡ Attack / Defense", "📊 Elo"],
        help="Switch between the two rating models. All tabs update."
    )
    use_ad = model_choice == "⚡ Attack / Defense"

    st.divider()
    st.subheader("Model Parameters")
    if use_ad:
        c1, c2 = st.columns(2)
        c1.metric("k_att",        str(AD_PARAMS["k_att"]))
        c1.metric("base_home",    str(AD_PARAMS["base_home"]))
        c2.metric("k_def",        str(AD_PARAMS["k_def"]))
        c2.metric("home_adv_log", str(AD_PARAMS["home_adv_log"]))
        st.caption(f"Brier: **0.600** | Baseline: 0.649")
    else:
        c1, c2 = st.columns(2)
        c1.metric("K-factor",      str(ELO_PARAMS["k"]))
        c1.metric("Goals Scale",   str(ELO_PARAMS["goals_scale"]))
        c2.metric("Home Adv",      f"+{ELO_PARAMS['home_advantage']}")
        c2.metric("Initial",       str(ELO_PARAMS["initial_rating"]))
        st.caption(f"Brier: **0.608** | Baseline: 0.649")

    st.divider()
    st.subheader("Data Coverage")
    st.write("**Seasons:** 2020-21 to 2025-26")
    st.write(f"**Matches:** {len(history_df):,}")
    st.write(f"**Last match:** {history_df['date'].max().strftime('%d %b %Y')}")

    st.divider()
    st.caption("Data: [football-data.co.uk](https://www.football-data.co.uk)")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🏆  Team Ratings", "🔮  Predictions", "📈  Rating Evolution", "📊  Model Performance"]
)

# ============================================================
# TAB 1 — Team Ratings
# ============================================================
with tab1:
    if use_ad:
        st.header("Team Ratings — Attack / Defense Model")
        st.caption("Separate attack and defense parameters, updated after every match.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top team (power)",  current_ratings.iloc[0]["team"],
                  f"{current_ratings.iloc[0]['power']:+.3f}")
        m2.metric("League avg power",  f"{current_ratings['power'].mean():+.3f}")
        m3.metric("Power spread",
                  f"{current_ratings.iloc[0]['power'] - current_ratings.iloc[-1]['power']:.3f}")
        m4.metric("Best attack",
                  current_ratings.loc[current_ratings["attack"].idxmax(), "team"],
                  f"{current_ratings['attack'].max():+.3f}")

        st.divider()

        # Scatter: attack vs defensive solidity
        st.subheader("Attack vs Defensive Solidity")
        st.caption("Each dot is a team. Top-right = Title Contenders. Bottom-left = Relegation Zone.")

        fig_s = px.scatter(
            current_ratings,
            x="attack", y="defensive_solidity",
            color="power", text="team",
            color_continuous_scale="RdYlGn",
            labels={"attack": "Attack", "defensive_solidity": "Defensive Solidity (–def)", "power": "Power"},
        )
        fig_s.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1, color="white")))
        med_a = current_ratings["attack"].median()
        med_d = current_ratings["defensive_solidity"].median()
        fig_s.add_hline(y=med_d, line_dash="dash", line_color="gray", opacity=0.5)
        fig_s.add_vline(x=med_a, line_dash="dash", line_color="gray", opacity=0.5)
        x_min = current_ratings["attack"].min() - 0.05
        x_max = current_ratings["attack"].max() + 0.05
        y_min = current_ratings["defensive_solidity"].min() - 0.05
        y_max = current_ratings["defensive_solidity"].max() + 0.05
        for label, x, y, anchor in [
            ("Title Contenders",        x_max, y_max, "right"),
            ("Clinical but Leaky",      x_max, y_min, "right"),
            ("Resilient but Toothless", x_min, y_max, "left"),
            ("Relegation Zone",         x_min, y_min, "left"),
        ]:
            fig_s.add_annotation(x=x, y=y, text=f"<i>{label}</i>",
                                 showarrow=False, font=dict(size=11, color="gray"), xanchor=anchor)
        fig_s.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20),
                            xaxis=dict(zeroline=True, zerolinecolor="lightgray"),
                            yaxis=dict(zeroline=True, zerolinecolor="lightgray"))
        st.plotly_chart(fig_s, use_container_width=True)

        st.divider()

        # Power bar chart
        st.subheader("Power Rankings")
        fig_b = px.bar(
            current_ratings.sort_values("power"),
            x="power", y="team", orientation="h",
            color="power", color_continuous_scale="RdYlGn",
            labels={"power": "Power (atk – def)", "team": ""},
            text="power",
        )
        fig_b.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_b.update_traces(texttemplate="%{text:+.3f}", textposition="outside")
        fig_b.update_layout(height=550, showlegend=False, coloraxis_showscale=False,
                            margin=dict(l=0, r=70, t=20, b=20))
        st.plotly_chart(fig_b, use_container_width=True)

        table = current_ratings[["team", "attack", "defensive_solidity", "power"]].copy()
        table.index.name = "Rank"
        table.columns = ["Team", "Attack", "Defensive Solidity", "Power"]
        st.dataframe(
            table.style.format({"Attack": "{:+.4f}", "Defensive Solidity": "{:+.4f}", "Power": "{:+.4f}"}),
            use_container_width=True,
        )

    else:
        # ---- ELO view ----
        st.header("Team Ratings — Elo Model")
        st.caption("Single composite rating, updated after every match. Baseline = 1500.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top team",     current_elo.iloc[0]["team"],
                  f"{current_elo.iloc[0]['elo_rating']:.0f}")
        m2.metric("League avg",   f"{current_elo['elo_rating'].mean():.0f}")
        m3.metric("Bottom team",  current_elo.iloc[-1]["team"],
                  f"{current_elo.iloc[-1]['elo_rating']:.0f}")
        m4.metric("Spread",
                  f"{current_elo.iloc[0]['elo_rating'] - current_elo.iloc[-1]['elo_rating']:.0f} pts")

        st.divider()

        fig_b = px.bar(
            current_elo.sort_values("elo_rating"),
            x="elo_rating", y="team", orientation="h",
            color="elo_rating", color_continuous_scale="RdYlGn",
            labels={"elo_rating": "Elo Rating", "team": ""},
            text="elo_rating",
        )
        fig_b.add_vline(x=1500, line_dash="dash", line_color="gray",
                        annotation_text="1500 baseline", annotation_position="top right")
        fig_b.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_b.update_layout(height=550, showlegend=False, coloraxis_showscale=False,
                            margin=dict(l=0, r=60, t=20, b=20),
                            xaxis_range=[current_elo["elo_rating"].min() - 30,
                                         current_elo["elo_rating"].max() + 60])
        st.plotly_chart(fig_b, use_container_width=True)

        table = current_elo[["team", "elo_rating"]].copy()
        table["vs 1500"] = table["elo_rating"].apply(lambda r: f"{r - 1500:+.0f}")
        table.index.name = "Rank"
        table.columns = ["Team", "Elo Rating", "vs Baseline"]
        st.dataframe(table, use_container_width=True)

# ============================================================
# TAB 2 — Predictions
# ============================================================
with tab2:
    if use_ad:
        hw_col, dw_col, aw_col = "home_win_prob", "draw_prob", "away_win_prob"
        xh_col, xa_col = "xg_home", "xg_away"
        tip_col = "most_likely_result"
        model_label = "Attack/Defense"
    else:
        hw_col, dw_col, aw_col = "elo_home_win_prob", "elo_draw_prob", "elo_away_win_prob"
        xh_col, xa_col = "elo_xg_home", "elo_xg_away"
        tip_col = "elo_most_likely_result"
        model_label = "Elo"

    st.header(f"Upcoming Match Predictions — {model_label} Model")

    if pred_df.empty:
        st.warning("No predictions found. Run `get_fixtures.py` and `predict_matches.py` first.")
    else:
        matchday = int(pred_df["matchday"].iloc[0])
        st.subheader(f"Matchday {matchday}")
        st.caption(f"Probabilities via Poisson model from current {model_label} ratings")

        # Stacked bar chart
        labels = [f"{r['home_team']} vs {r['away_team']}" for _, r in pred_df.iterrows()]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Home win", y=labels, x=pred_df[hw_col] * 100, orientation="h",
            marker_color="#2ecc71",
            text=[f"{p*100:.0f}%" for p in pred_df[hw_col]], textposition="inside",
        ))
        fig.add_trace(go.Bar(
            name="Draw", y=labels, x=pred_df[dw_col] * 100, orientation="h",
            marker_color="#95a5a6",
            text=[f"{p*100:.0f}%" for p in pred_df[dw_col]], textposition="inside",
        ))
        fig.add_trace(go.Bar(
            name="Away win", y=labels, x=pred_df[aw_col] * 100, orientation="h",
            marker_color="#e74c3c",
            text=[f"{p*100:.0f}%" for p in pred_df[aw_col]], textposition="inside",
        ))
        fig.update_layout(
            barmode="stack", height=420,
            xaxis=dict(title="Probability (%)", range=[0, 100]),
            yaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=0, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Match Details")

        for _, row in pred_df.iterrows():
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 3])
            if use_ad:
                c1.markdown(
                    f"**{row['home_team']}**  \n"
                    f"*Atk: {row['home_attack']:+.3f} | "
                    f"Def: {-row['home_defense']:+.3f} | "
                    f"Power: {row['home_power']:+.3f}*"
                )
                c5.markdown(
                    f"**{row['away_team']}**  \n"
                    f"*Atk: {row['away_attack']:+.3f} | "
                    f"Def: {-row['away_defense']:+.3f} | "
                    f"Power: {row['away_power']:+.3f}*  \n"
                    f"xG: {row[xh_col]:.1f}–{row[xa_col]:.1f}"
                )
            else:
                c1.markdown(
                    f"**{row['home_team']}**  \n"
                    f"*Elo: {row['elo_home_rating']:.0f}*"
                )
                c5.markdown(
                    f"**{row['away_team']}**  \n"
                    f"*Elo: {row['elo_away_rating']:.0f}*  \n"
                    f"xG: {row[xh_col]:.1f}–{row[xa_col]:.1f}"
                )
            c2.metric("H", f"{row[hw_col]*100:.0f}%")
            c3.metric("D", f"{row[dw_col]*100:.0f}%")
            c4.metric("A", f"{row[aw_col]*100:.0f}%")
            st.divider()

# ============================================================
# TAB 3 — Rating Evolution
# ============================================================
with tab3:
    if use_ad:
        st.header("Rating Evolution — Attack / Defense")
        all_teams = sorted(ad_ts_df["team"].unique())
        top5 = current_ratings["team"].head(5).tolist()

        col_sel, col_met = st.columns([3, 1])
        with col_sel:
            selected = st.multiselect("Teams:", options=all_teams, default=top5)
        with col_met:
            metrics = st.multiselect(
                "Metrics:", options=["attack", "defensive_solidity", "power"], default=["power"]
            )

        if selected and metrics:
            filtered = ad_ts_df[ad_ts_df["team"].isin(selected) & ad_ts_df["metric"].isin(metrics)]
            metric_labels = {"attack": "Attack", "defensive_solidity": "Def Solidity", "power": "Power"}
            if len(metrics) == 1:
                fig = px.line(filtered, x="date", y="value", color="team",
                              labels={"date": "", "value": metric_labels[metrics[0]], "team": "Team"},
                              title=f"{metric_labels[metrics[0]]} Over Time")
                fig.add_hline(y=0, line_dash="dash", line_color="lightgray")
            else:
                filtered = filtered.copy()
                filtered["metric_label"] = filtered["metric"].map(metric_labels)
                fig = px.line(filtered, x="date", y="value", color="team", facet_row="metric_label",
                              labels={"date": "", "value": "Value", "team": "Team"},
                              title="Rating Evolution by Metric")
                fig.update_yaxes(matches=None)
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig.update_layout(
                height=500 if len(metrics) == 1 else 200 * len(metrics),
                hovermode="x unified", margin=dict(l=0, r=0, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            snap = current_ratings[current_ratings["team"].isin(selected)].sort_values("power", ascending=False)
            snap = snap.reset_index(drop=True)
            snap.index = range(1, len(snap) + 1)
            snap.index.name = "Rank"
            st.dataframe(
                snap[["team", "attack", "defensive_solidity", "power"]].rename(columns={
                    "team": "Team", "attack": "Attack",
                    "defensive_solidity": "Def Solidity", "power": "Power",
                }).style.format({"Attack": "{:+.4f}", "Def Solidity": "{:+.4f}", "Power": "{:+.4f}"}),
                use_container_width=True,
            )
        else:
            st.info("Select at least one team and one metric.")

    else:
        st.header("Rating Evolution — Elo")
        all_teams = sorted(elo_ts_df["team"].unique())
        top5 = current_elo["team"].head(5).tolist()
        selected = st.multiselect("Teams:", options=all_teams, default=top5)

        if selected:
            filtered = elo_ts_df[elo_ts_df["team"].isin(selected)]
            fig = px.line(filtered, x="date", y="value", color="team",
                          labels={"date": "", "value": "Elo Rating", "team": "Team"},
                          title="Elo Rating Over Time")
            fig.add_hline(y=1500, line_dash="dash", line_color="lightgray",
                          annotation_text="1500 baseline")
            fig.update_layout(height=500, hovermode="x unified", margin=dict(l=0, r=0, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            snap = current_elo[current_elo["team"].isin(selected)].copy()
            snap = snap.reset_index(drop=True)
            snap.index = range(1, len(snap) + 1)
            snap.index.name = "Rank"
            snap["vs 1500"] = snap["elo_rating"].apply(lambda r: f"{r - 1500:+.0f}")
            st.dataframe(
                snap[["team", "elo_rating", "vs 1500"]].rename(
                    columns={"team": "Team", "elo_rating": "Elo Rating", "vs 1500": "vs Baseline"}
                ),
                use_container_width=True,
            )
        else:
            st.info("Select at least one team.")

# ============================================================
# TAB 4 — Model Performance
# ============================================================
with tab4:
    if use_ad:
        st.header("Model Performance — Attack / Defense")
        st.caption("Backtesting: outcome predicted using pre-match AD ratings only.")

        with st.spinner("Computing metrics ..."):
            m = compute_ad_backtest_metrics(history_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%",
                  delta=f"{(m['accuracy'] - m['baseline_accuracy'])*100:+.1f}% vs always-H",
                  delta_color="normal")
        c2.metric("Baseline Accuracy", f"{m['baseline_accuracy']*100:.1f}%")
        c3.metric("Brier Score", f"{m['brier']:.4f}",
                  delta=f"{m['brier'] - m['brier_baseline']:+.4f} vs baseline",
                  delta_color="inverse")
        c4.metric("Brier Baseline", f"{m['brier_baseline']:.4f}")

        st.divider()
        st.subheader("Goal Prediction Accuracy (RMSE)")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("RMSE Home Goals", f"{m['rmse_home']:.3f}",
                  delta=f"{m['rmse_home'] - m['rmse_naive_home']:+.3f} vs naive", delta_color="inverse")
        g2.metric("RMSE Away Goals", f"{m['rmse_away']:.3f}",
                  delta=f"{m['rmse_away'] - m['rmse_naive_away']:+.3f} vs naive", delta_color="inverse")
        g3.metric("Naive RMSE (home)", f"{m['rmse_naive_home']:.3f}")
        g4.metric("Naive RMSE (away)", f"{m['rmse_naive_away']:.3f}")

        st.divider()
        st.subheader("xG vs Actual Goals")
        col_h, col_a = st.columns(2)
        for col, xg, actual, label in [
            (col_h, m["xg_h"], m["actual_hg"], "Home"),
            (col_a, m["xg_a"], m["actual_ag"], "Away"),
        ]:
            with col:
                st.caption(f"{label}: predicted xG vs actual goals")
                sdf = pd.DataFrame({"xg": xg, "actual": actual})
                fig = px.scatter(sdf, x="xg", y="actual", opacity=0.25,
                                 labels={"xg": f"Predicted xG ({label.lower()})",
                                         "actual": f"Actual goals ({label.lower()})"})
                mx = max(sdf["xg"].max(), sdf["actual"].max()) + 0.5
                fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx,
                              line=dict(color="red", dash="dash"))
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=20))
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        sacc = season_accuracy_ad(history_df)

    else:
        st.header("Model Performance — Elo")
        st.caption("Backtesting: outcome predicted using pre-match Elo ratings only.")

        with st.spinner("Computing metrics ..."):
            m = compute_elo_backtest_metrics(history_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%",
                  delta=f"{(m['accuracy'] - m['baseline_accuracy'])*100:+.1f}% vs always-H",
                  delta_color="normal")
        c2.metric("Baseline Accuracy", f"{m['baseline_accuracy']*100:.1f}%")
        c3.metric("Brier Score", f"{m['brier']:.4f}",
                  delta=f"{m['brier'] - m['brier_baseline']:+.4f} vs baseline",
                  delta_color="inverse")
        c4.metric("Brier Baseline", f"{m['brier_baseline']:.4f}")

        st.divider()
        sacc = season_accuracy_elo(history_df)

    # Calibration chart (shared structure, data differs)
    st.subheader("Calibration: Home Win Predictions")
    st.caption("Bars should match the red diagonal for a well-calibrated model.")
    cal = m["calibration"]
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Bar(
        x=cal["predicted_range"], y=cal["actual_rate"], name="Actual rate",
        marker_color="#3498db",
        text=[f"{v:.0%}<br>n={n}" for v, n in zip(cal["actual_rate"], cal["count"])],
        textposition="outside",
    ))
    mid_points = [i * 0.1 + 0.05 for i in range(len(cal))]
    fig_cal.add_trace(go.Scatter(
        x=cal["predicted_range"], y=mid_points[:len(cal)],
        mode="lines+markers", name="Perfect calibration",
        line=dict(color="red", dash="dash"),
    ))
    fig_cal.update_layout(
        xaxis_title="Predicted home-win probability range",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=20),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # Season accuracy chart (shared)
    st.subheader("Accuracy by Season")
    sacc["season_label"] = sacc["season"].apply(_season_label)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=sacc["season_label"], y=sacc["accuracy"], name="Model accuracy",
        marker_color="#2ecc71",
        text=[f"{v:.1%}" for v in sacc["accuracy"]], textposition="outside",
    ))
    fig2.add_trace(go.Bar(
        x=sacc["season_label"], y=sacc["baseline"], name="Baseline (always H)",
        marker_color="#bdc3c7",
        text=[f"{v:.1%}" for v in sacc["baseline"]], textposition="outside",
    ))
    fig2.update_layout(
        barmode="group", yaxis=dict(tickformat=".0%", range=[0, 0.75]),
        height=360, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(
        sacc[["season_label", "matches", "accuracy", "baseline"]].rename(columns={
            "season_label": "Season", "matches": "Matches",
            "accuracy": "Model Accuracy", "baseline": "Baseline (always H)",
        }).style.format({"Model Accuracy": "{:.1%}", "Baseline (always H)": "{:.1%}"}),
        use_container_width=True,
    )
