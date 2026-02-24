"""
dashboard.py — Phase 4, Step 13

Streamlit dashboard for the Bundesliga SPI Prediction Model (Attack/Defense).
Run with: streamlit run src/dashboard.py
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
# Data loaders (all cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_current_ratings() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / "data/processed/current_ratings.csv")


@st.cache_data
def load_history() -> pd.DataFrame:
    return pd.read_csv(
        BASE_DIR / "data/processed/team_ratings_history.csv",
        parse_dates=["date"],
    )


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = BASE_DIR / "data/predictions/upcoming_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def build_ad_timeseries(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-format dataframe with attack, defense (raw), and power per team per match.
    Returns columns: date, team, metric, value
    where metric ∈ {attack, defensive_solidity, power}
    """
    rows = []
    for col, label in [
        ("home_atk_post",  "attack"),
        ("away_atk_post",  "attack"),
    ]:
        team_col = "home_team" if "home" in col else "away_team"
        sub = history_df[["date", team_col, col]].copy()
        sub.columns = ["date", "team", "value"]
        sub["metric"] = label
        rows.append(sub)

    # defensive_solidity = -def_post
    for col, label in [
        ("home_def_post", "defensive_solidity"),
        ("away_def_post", "defensive_solidity"),
    ]:
        team_col = "home_team" if "home" in col else "away_team"
        sub = history_df[["date", team_col, col]].copy()
        sub.columns = ["date", "team", "value"]
        sub["value"] = -sub["value"]   # flip sign: higher = better defense
        sub["metric"] = label
        rows.append(sub)

    # power = atk - def_
    home_power = history_df[["date", "home_team", "home_atk_post", "home_def_post"]].copy()
    home_power["value"] = home_power["home_atk_post"] - home_power["home_def_post"]
    home_power["metric"] = "power"
    home_power = home_power[["date", "home_team", "value", "metric"]].rename(
        columns={"home_team": "team"}
    )

    away_power = history_df[["date", "away_team", "away_atk_post", "away_def_post"]].copy()
    away_power["value"] = away_power["away_atk_post"] - away_power["away_def_post"]
    away_power["metric"] = "power"
    away_power = away_power[["date", "away_team", "value", "metric"]].rename(
        columns={"away_team": "team"}
    )

    rows.append(home_power)
    rows.append(away_power)

    ts = pd.concat(rows).sort_values(["team", "metric", "date"]).reset_index(drop=True)
    return ts


@st.cache_data
def compute_backtest_metrics(history_df: pd.DataFrame) -> dict:
    """
    For every match, predict outcome using pre-match AD ratings,
    then compare to actual result.
    Returns accuracy, Brier score, calibration data, and goal RMSE.
    """
    from model_config import AD_PARAMS

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

    # Vectorised Poisson PMF: shape (N, 11)
    goals = np.arange(11, dtype=float)

    def pmf_batch(xg: np.ndarray) -> np.ndarray:
        log_p = goals * np.log(xg[:, None]) - xg[:, None] - gammaln(goals + 1)
        return np.exp(log_p)

    pmf_h_arr = pmf_batch(xg_h)
    pmf_a_arr = pmf_batch(xg_a)
    joint = pmf_h_arr[:, :, None] * pmf_a_arr[:, None, :]  # (N, 11, 11)

    mask_hw = np.tril(np.ones((11, 11), dtype=bool), k=-1)
    mask_d  = np.eye(11, dtype=bool)
    mask_aw = np.triu(np.ones((11, 11), dtype=bool), k=1)

    ph  = joint[:, mask_hw].sum(axis=1)
    pd_ = joint[:, mask_d].sum(axis=1)
    pa  = joint[:, mask_aw].sum(axis=1)
    tot = ph + pd_ + pa
    ph /= tot; pd_ /= tot; pa /= tot

    oh = (results == "H").astype(float)
    od = (results == "D").astype(float)
    oa = (results == "A").astype(float)

    brier = float(np.mean((ph - oh)**2 + (pd_ - od)**2 + (pa - oa)**2))
    brier_baseline = float(np.mean((1 - oh)**2 + od**2 + oa**2))

    pred_label = np.where(
        (ph >= pd_) & (ph >= pa), "H",
        np.where(pd_ >= pa, "D", "A")
    )
    accuracy = float(np.mean(pred_label == results))
    baseline_accuracy = float(np.mean(results == "H"))

    # Calibration bins
    bins = np.arange(0, 1.1, 0.1)
    bucket_labels, actual_rates, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (ph >= lo) & (ph < hi)
        if mask.sum() > 0:
            bucket_labels.append(f"{lo:.0%}–{hi:.0%}")
            actual_rates.append(oh[mask].mean())
            counts.append(mask.sum())

    # Goal RMSE vs naive baseline
    actual_hg = history_df["home_goals"].values.astype(float)
    actual_ag = history_df["away_goals"].values.astype(float)
    rmse_home = float(np.sqrt(np.mean((xg_h - actual_hg) ** 2)))
    rmse_away = float(np.sqrt(np.mean((xg_a - actual_ag) ** 2)))
    rmse_naive_home = float(np.sqrt(np.mean((base_home - actual_hg) ** 2)))
    rmse_naive_away = float(np.sqrt(np.mean((base_away - actual_ag) ** 2)))

    # Goal scatter data (sample for performance)
    scatter_df = pd.DataFrame({
        "xg_home":      xg_h,
        "xg_away":      xg_a,
        "actual_home":  actual_hg,
        "actual_away":  actual_ag,
    })

    return {
        "accuracy":          accuracy,
        "baseline_accuracy": baseline_accuracy,
        "brier":             brier,
        "brier_baseline":    brier_baseline,
        "n_matches":         len(results),
        "rmse_home":         rmse_home,
        "rmse_away":         rmse_away,
        "rmse_naive_home":   rmse_naive_home,
        "rmse_naive_away":   rmse_naive_away,
        "xg_h":              xg_h,
        "actual_hg":         actual_hg,
        "xg_a":              xg_a,
        "actual_ag":         actual_ag,
        "calibration": pd.DataFrame({
            "predicted_range": bucket_labels,
            "actual_rate":     actual_rates,
            "count":           counts,
        }),
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ratings_df = load_current_ratings()
history_df = load_history()
pred_df    = load_predictions()
ts_df      = build_ad_timeseries(history_df)

# Current season's 18 teams
current_teams = set(
    history_df[history_df["season"] == history_df["season"].max()]["home_team"].unique()
) | set(
    history_df[history_df["season"] == history_df["season"].max()]["away_team"].unique()
)
current_ratings = ratings_df[ratings_df["team"].isin(current_teams)].copy()
current_ratings = current_ratings.sort_values("power", ascending=False).reset_index(drop=True)
current_ratings.index = range(1, len(current_ratings) + 1)

# Add defensive_solidity column for display
current_ratings["defensive_solidity"] = -current_ratings["defense"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
from model_config import AD_PARAMS

with st.sidebar:
    st.title("⚽ Bundesliga SPI")
    st.caption("FiveThirtyEight-style prediction model")
    st.divider()

    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)
    col1.metric("k_att", str(AD_PARAMS["k_att"]))
    col1.metric("base_home", str(AD_PARAMS["base_home"]))
    col2.metric("k_def", str(AD_PARAMS["k_def"]))
    col2.metric("home_adv_log", str(AD_PARAMS["home_adv_log"]))
    st.caption("Attack/Defense model | Optimised via Brier score grid search")

    st.divider()
    st.subheader("Data Coverage")
    st.write(f"**Seasons:** 2020-21 to 2025-26")
    st.write(f"**Matches:** {len(history_df):,}")
    st.write(f"**Last match:** {history_df['date'].max().strftime('%d %b %Y')}")

    st.divider()
    st.caption("Data: [football-data.co.uk](https://www.football-data.co.uk)")
    st.caption(
        "Method: [FiveThirtyEight SPI]"
        "(https://fivethirtyeight.com/methodology/how-our-club-soccer-predictions-work/)"
    )

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
    st.header("Current Bundesliga Team Ratings")
    st.caption("Attack/Defense ratings after all completed matches (2020-21 through 2025-26)")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top team (power)", current_ratings.iloc[0]["team"],
              f"{current_ratings.iloc[0]['power']:+.3f}")
    m2.metric("League avg power", f"{current_ratings['power'].mean():+.3f}")
    m3.metric("Power spread",
              f"{current_ratings.iloc[0]['power'] - current_ratings.iloc[-1]['power']:.3f}")
    m4.metric("Best attack",
              current_ratings.loc[current_ratings["attack"].idxmax(), "team"],
              f"{current_ratings['attack'].max():+.3f}")

    st.divider()

    # Attack vs Defensive Solidity scatter plot
    st.subheader("Attack vs Defensive Solidity")
    st.caption(
        "x-axis: attack rating (higher = scores more than expected)  |  "
        "y-axis: defensive solidity = –defense (higher = concedes less than expected)"
    )

    scatter_data = current_ratings.copy()

    fig_scatter = px.scatter(
        scatter_data,
        x="attack",
        y="defensive_solidity",
        color="power",
        text="team",
        color_continuous_scale="RdYlGn",
        labels={
            "attack":             "Attack Rating",
            "defensive_solidity": "Defensive Solidity (–def)",
            "power":              "Power",
        },
        size_max=14,
    )
    fig_scatter.update_traces(
        textposition="top center",
        marker=dict(size=12, line=dict(width=1, color="white")),
    )

    # Quadrant lines at league medians
    med_atk = scatter_data["attack"].median()
    med_def = scatter_data["defensive_solidity"].median()

    fig_scatter.add_hline(y=med_def, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.add_vline(x=med_atk, line_dash="dash", line_color="gray", opacity=0.5)

    # Quadrant labels
    x_min = scatter_data["attack"].min() - 0.05
    x_max = scatter_data["attack"].max() + 0.05
    y_min = scatter_data["defensive_solidity"].min() - 0.05
    y_max = scatter_data["defensive_solidity"].max() + 0.05

    for label, x, y, anchor in [
        ("Title Contenders",       x_max, y_max, "right"),
        ("Clinical but Leaky",     x_max, y_min, "right"),
        ("Resilient but Toothless", x_min, y_max, "left"),
        ("Relegation Zone",        x_min, y_min, "left"),
    ]:
        fig_scatter.add_annotation(
            x=x, y=y, text=f"<i>{label}</i>",
            showarrow=False, font=dict(size=11, color="gray"),
            xanchor=anchor,
        )

    fig_scatter.update_layout(
        height=520,
        coloraxis_colorbar=dict(title="Power"),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(zeroline=True, zerolinecolor="lightgray"),
        yaxis=dict(zeroline=True, zerolinecolor="lightgray"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # Power bar chart
    st.subheader("Power Rankings")
    fig_bar = px.bar(
        current_ratings.sort_values("power"),
        x="power",
        y="team",
        orientation="h",
        color="power",
        color_continuous_scale="RdYlGn",
        labels={"power": "Power (atk – def)", "team": ""},
        text="power",
    )
    fig_bar.add_vline(x=0, line_dash="dash", line_color="gray",
                      annotation_text="0 baseline", annotation_position="top right")
    fig_bar.update_traces(texttemplate="%{text:+.3f}", textposition="outside")
    fig_bar.update_layout(
        height=550,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=0, r=70, t=20, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Data table
    table = current_ratings[["team", "attack", "defensive_solidity", "power"]].copy()
    table.index.name = "Rank"
    table.columns = ["Team", "Attack", "Defensive Solidity", "Power"]
    st.dataframe(
        table.style.format({"Attack": "{:+.4f}", "Defensive Solidity": "{:+.4f}", "Power": "{:+.4f}"}),
        use_container_width=True,
    )

# ============================================================
# TAB 2 — Predictions
# ============================================================
with tab2:
    st.header("Upcoming Match Predictions")

    if pred_df.empty:
        st.warning("No predictions found. Run `get_fixtures.py` and `predict_matches.py` first.")
    else:
        matchday = int(pred_df["matchday"].iloc[0])
        st.subheader(f"Matchday {matchday}")
        st.caption("Probabilities via Poisson model from current Attack/Defense ratings")

        # Stacked probability chart
        labels = [f"{r['home_team']} vs {r['away_team']}" for _, r in pred_df.iterrows()]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Home win",
            y=labels,
            x=pred_df["home_win_prob"] * 100,
            orientation="h",
            marker_color="#2ecc71",
            text=[f"{p*100:.0f}%" for p in pred_df["home_win_prob"]],
            textposition="inside",
        ))
        fig.add_trace(go.Bar(
            name="Draw",
            y=labels,
            x=pred_df["draw_prob"] * 100,
            orientation="h",
            marker_color="#95a5a6",
            text=[f"{p*100:.0f}%" for p in pred_df["draw_prob"]],
            textposition="inside",
        ))
        fig.add_trace(go.Bar(
            name="Away win",
            y=labels,
            x=pred_df["away_win_prob"] * 100,
            orientation="h",
            marker_color="#e74c3c",
            text=[f"{p*100:.0f}%" for p in pred_df["away_win_prob"]],
            textposition="inside",
        ))
        fig.update_layout(
            barmode="stack",
            height=420,
            xaxis=dict(title="Probability (%)", range=[0, 100]),
            yaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=0, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Detailed match cards
        st.subheader("Match Details")
        for _, row in pred_df.iterrows():
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 3])
            c1.markdown(
                f"**{row['home_team']}**  \n"
                f"*Atk: {row['home_attack']:+.3f} | "
                f"Def sol: {-row['home_defense']:+.3f} | "
                f"Power: {row['home_power']:+.3f}*"
            )
            c2.metric("H", f"{row['home_win_prob']*100:.0f}%")
            c3.metric("D", f"{row['draw_prob']*100:.0f}%")
            c4.metric("A", f"{row['away_win_prob']*100:.0f}%")
            c5.markdown(
                f"**{row['away_team']}**  \n"
                f"*Atk: {row['away_attack']:+.3f} | "
                f"Def sol: {-row['away_defense']:+.3f} | "
                f"Power: {row['away_power']:+.3f}*  \n"
                f"xG: {row['xg_home']:.1f}–{row['xg_away']:.1f}"
            )
            st.divider()

# ============================================================
# TAB 3 — Rating Evolution
# ============================================================
with tab3:
    st.header("Team Rating Evolution")
    st.caption("Track attack, defensive solidity, and power over time")

    top5 = current_ratings["team"].head(5).tolist()
    all_team_list = sorted(ts_df["team"].unique())

    col_sel, col_metric = st.columns([3, 1])
    with col_sel:
        selected = st.multiselect(
            "Select teams to display:",
            options=all_team_list,
            default=top5,
        )
    with col_metric:
        selected_metrics = st.multiselect(
            "Metrics:",
            options=["attack", "defensive_solidity", "power"],
            default=["power"],
        )

    if not selected:
        st.info("Select at least one team above.")
    elif not selected_metrics:
        st.info("Select at least one metric.")
    else:
        filtered = ts_df[
            ts_df["team"].isin(selected) & ts_df["metric"].isin(selected_metrics)
        ]

        metric_labels = {
            "attack":             "Attack Rating",
            "defensive_solidity": "Defensive Solidity",
            "power":              "Power (atk – def)",
        }

        if len(selected_metrics) == 1:
            fig = px.line(
                filtered,
                x="date",
                y="value",
                color="team",
                labels={"date": "", "value": metric_labels[selected_metrics[0]], "team": "Team"},
                title=f"{metric_labels[selected_metrics[0]]} Over Time",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="lightgray",
                          annotation_text="0 baseline")
        else:
            # Faceted by metric
            filtered["metric_label"] = filtered["metric"].map(metric_labels)
            fig = px.line(
                filtered,
                x="date",
                y="value",
                color="team",
                facet_row="metric_label",
                labels={"date": "", "value": "Value", "team": "Team"},
                title="Rating Evolution by Metric",
            )
            fig.update_yaxes(matches=None)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        fig.update_layout(
            height=500 if len(selected_metrics) == 1 else 200 * len(selected_metrics),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Snapshot table for selected teams
        snap = current_ratings[current_ratings["team"].isin(selected)].copy()
        snap = snap.sort_values("power", ascending=False).reset_index(drop=True)
        snap.index = range(1, len(snap) + 1)
        snap.index.name = "Rank"
        st.dataframe(
            snap[["team", "attack", "defensive_solidity", "power"]].rename(columns={
                "team": "Team", "attack": "Attack",
                "defensive_solidity": "Def Solidity", "power": "Power",
            }).style.format({
                "Attack": "{:+.4f}", "Def Solidity": "{:+.4f}", "Power": "{:+.4f}"
            }),
            use_container_width=True,
        )

# ============================================================
# TAB 4 — Model Performance
# ============================================================
with tab4:
    st.header("Model Performance")
    st.caption(
        "Backtesting: for each match, outcome predicted using pre-match AD ratings only."
    )

    with st.spinner("Computing backtest metrics ..."):
        metrics = compute_backtest_metrics(history_df)

    # Summary metrics row 1: outcome prediction
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Outcome Accuracy",
        f"{metrics['accuracy']*100:.1f}%",
        delta=f"{(metrics['accuracy'] - metrics['baseline_accuracy'])*100:+.1f}% vs always-H",
        delta_color="normal",
    )
    m2.metric(
        "Baseline Accuracy",
        f"{metrics['baseline_accuracy']*100:.1f}%",
        help="Accuracy if we always predict home win",
    )
    m3.metric(
        "Brier Score",
        f"{metrics['brier']:.4f}",
        delta=f"{metrics['brier'] - metrics['brier_baseline']:+.4f} vs baseline",
        delta_color="inverse",
        help="Lower = better. Multi-class Brier score, max = 2.0",
    )
    m4.metric(
        "Brier Baseline",
        f"{metrics['brier_baseline']:.4f}",
        help="Brier score if always predicting home win",
    )

    st.divider()

    # Summary metrics row 2: goal prediction
    st.subheader("Goal Prediction Accuracy (RMSE)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(
        "RMSE Home Goals",
        f"{metrics['rmse_home']:.3f}",
        delta=f"{metrics['rmse_home'] - metrics['rmse_naive_home']:+.3f} vs naive",
        delta_color="inverse",
        help="Root mean squared error: predicted xG_home vs actual home goals",
    )
    g2.metric(
        "RMSE Away Goals",
        f"{metrics['rmse_away']:.3f}",
        delta=f"{metrics['rmse_away'] - metrics['rmse_naive_away']:+.3f} vs naive",
        delta_color="inverse",
        help="Root mean squared error: predicted xG_away vs actual away goals",
    )
    g3.metric(
        "Naive RMSE (home)",
        f"{metrics['rmse_naive_home']:.3f}",
        help=f"Always predict {AD_PARAMS['base_home']:.2f} home goals",
    )
    g4.metric(
        "Naive RMSE (away)",
        f"{metrics['rmse_naive_away']:.3f}",
        help=f"Always predict {AD_PARAMS['base_away']:.2f} away goals",
    )

    st.divider()

    # Goal prediction scatter
    st.subheader("xG vs Actual Goals")
    col_h, col_a = st.columns(2)

    with col_h:
        st.caption("Home team: predicted xG_home vs actual goals scored")
        scatter_home = pd.DataFrame({
            "xg": metrics["xg_h"],
            "actual": metrics["actual_hg"],
        })
        fig_gh = px.scatter(
            scatter_home, x="xg", y="actual",
            opacity=0.25,
            labels={"xg": "Predicted xG (home)", "actual": "Actual goals (home)"},
        )
        max_val = max(scatter_home["xg"].max(), scatter_home["actual"].max()) + 0.5
        fig_gh.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash"),
        )
        fig_gh.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=20))
        st.plotly_chart(fig_gh, use_container_width=True)

    with col_a:
        st.caption("Away team: predicted xG_away vs actual goals scored")
        scatter_away = pd.DataFrame({
            "xg": metrics["xg_a"],
            "actual": metrics["actual_ag"],
        })
        fig_ga = px.scatter(
            scatter_away, x="xg", y="actual",
            opacity=0.25,
            labels={"xg": "Predicted xG (away)", "actual": "Actual goals (away)"},
        )
        max_val = max(scatter_away["xg"].max(), scatter_away["actual"].max()) + 0.5
        fig_ga.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash"),
        )
        fig_ga.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=20))
        st.plotly_chart(fig_ga, use_container_width=True)

    st.divider()

    # Calibration chart
    st.subheader("Calibration: Home Win Predictions")
    st.caption(
        "A well-calibrated model has bars matching the diagonal. "
        "Each bar shows the actual home-win rate for matches where we predicted that probability range."
    )
    cal = metrics["calibration"]
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Bar(
        x=cal["predicted_range"],
        y=cal["actual_rate"],
        name="Actual rate",
        marker_color="#3498db",
        text=[f"{v:.0%}<br>n={n}" for v, n in zip(cal["actual_rate"], cal["count"])],
        textposition="outside",
    ))
    mid_points = [i * 0.1 + 0.05 for i in range(len(cal))]
    fig_cal.add_trace(go.Scatter(
        x=cal["predicted_range"],
        y=mid_points[:len(cal)],
        mode="lines+markers",
        name="Perfect calibration",
        line=dict(color="red", dash="dash"),
    ))
    fig_cal.update_layout(
        xaxis_title="Predicted home-win probability range",
        yaxis_title="Actual home-win rate",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=20),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # Season-by-season accuracy
    st.subheader("Accuracy by Season")

    @st.cache_data
    def season_accuracy(history_df: pd.DataFrame) -> pd.DataFrame:
        from model_config import AD_PARAMS
        home_adv_log = AD_PARAMS["home_adv_log"]
        base_home    = AD_PARAMS["base_home"]
        base_away    = AD_PARAMS["base_away"]

        rows = []
        for season, grp in history_df.groupby("season"):
            atk_h = grp["home_atk_pre"].values
            def_h = grp["home_def_pre"].values
            atk_a = grp["away_atk_pre"].values
            def_a = grp["away_def_pre"].values

            exp_h = np.clip(atk_h + def_a + home_adv_log, -3.0, 3.0)
            exp_a = np.clip(atk_a + def_h,                -3.0, 3.0)
            xg_h  = base_home * np.exp(exp_h)
            xg_a  = base_away * np.exp(exp_a)

            goals = np.arange(11, dtype=float)
            def pmf_batch(xg):
                log_p = goals * np.log(xg[:, None]) - xg[:, None] - gammaln(goals + 1)
                return np.exp(log_p)

            pmf_h_arr = pmf_batch(xg_h)
            pmf_a_arr = pmf_batch(xg_a)
            joint = pmf_h_arr[:, :, None] * pmf_a_arr[:, None, :]

            mask_hw = np.tril(np.ones((11, 11), dtype=bool), k=-1)
            mask_d  = np.eye(11, dtype=bool)
            mask_aw = np.triu(np.ones((11, 11), dtype=bool), k=1)

            ph  = joint[:, mask_hw].sum(axis=1)
            pd2 = joint[:, mask_d].sum(axis=1)
            pa  = joint[:, mask_aw].sum(axis=1)
            tot = ph + pd2 + pa
            ph /= tot; pd2 /= tot; pa /= tot

            actual = grp["result"].values
            pred_label = np.where(
                (ph >= pd2) & (ph >= pa), "H",
                np.where(pd2 >= pa, "D", "A")
            )
            acc  = float(np.mean(pred_label == actual))
            base = float(np.mean(actual == "H"))
            rows.append({"season": season, "accuracy": acc, "baseline": base,
                         "matches": len(grp)})
        return pd.DataFrame(rows)

    sacc = season_accuracy(history_df)
    sacc["season_label"] = sacc["season"].apply(lambda s: f"20{s[:2]}-{s[2:]}")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=sacc["season_label"], y=sacc["accuracy"],
        name="Model accuracy", marker_color="#2ecc71",
        text=[f"{v:.1%}" for v in sacc["accuracy"]],
        textposition="outside",
    ))
    fig2.add_trace(go.Bar(
        x=sacc["season_label"], y=sacc["baseline"],
        name="Baseline (always H)", marker_color="#bdc3c7",
        text=[f"{v:.1%}" for v in sacc["baseline"]],
        textposition="outside",
    ))
    fig2.update_layout(
        barmode="group",
        yaxis=dict(tickformat=".0%", range=[0, 0.75]),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
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
