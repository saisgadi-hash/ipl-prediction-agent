"""
IPL Prediction Agent — Streamlit Dashboard v2.0
Apple Design Award-level UI with SHAP justifications and user feedback.

HOW TO RUN:
    streamlit run src/dashboard/app.py
"""

import os
import sys
import uuid

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data_collection.team_name_mapper import (
    standardise_team_name, get_short_code, ACTIVE_TEAMS, TEAM_SHORT_CODES,
)

# ── Page Config ──
st.set_page_config(
    page_title="IPL Prediction Agent",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Team Colors (official IPL team colors) ──
TEAM_COLORS = {
    "Chennai Super Kings": "#FFCB05",
    "Mumbai Indians": "#004BA0",
    "Royal Challengers Bengaluru": "#EC1C24",
    "Kolkata Knight Riders": "#3A225D",
    "Delhi Capitals": "#004C93",
    "Punjab Kings": "#ED1B24",
    "Rajasthan Royals": "#EA1A85",
    "Sunrisers Hyderabad": "#FF822A",
    "Lucknow Super Giants": "#A72056",
    "Gujarat Titans": "#1C1C1C",
    "Rising Pune Supergiant": "#6F2C91",
    "Pune Warriors": "#2F9BE3",
    "Deccan Chargers": "#B0B0B0",
    "Kochi Tuskers Kerala": "#6F2C91",
    "Gujarat Lions": "#E04F16",
}

# Secondary accent colors for gradients
TEAM_COLORS_LIGHT = {
    "Chennai Super Kings": "#FFF3C4",
    "Mumbai Indians": "#A8D0F0",
    "Royal Challengers Bengaluru": "#F8A0A4",
    "Kolkata Knight Riders": "#C4A8E0",
    "Delhi Capitals": "#A0C4E8",
    "Punjab Kings": "#F8A0A4",
    "Rajasthan Royals": "#F0A0D0",
    "Sunrisers Hyderabad": "#FFD0A0",
    "Lucknow Super Giants": "#D0A0B8",
    "Gujarat Titans": "#808080",
}


def get_team_color(name: str) -> str:
    return TEAM_COLORS.get(standardise_team_name(name), "#888888")


TEAM_SHORT = TEAM_SHORT_CODES


# ══════════════════════════════════════════
# CUSTOM CSS — Clean, Modern, Minimal Design
# ══════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global Reset ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    /* ── Background ── */
    .stApp {
        background: #0B0E17;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0F1220;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stRadio label {
        color: #B0B8C8;
    }

    /* ── Typography ── */
    h1 {
        color: #FFFFFF;
        font-weight: 800;
        letter-spacing: -0.03em;
        font-size: 2.2rem !important;
    }
    h2 {
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    h3 {
        color: #E0E4EC;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    /* ── Metric Cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #141824, #1A1F32);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 20px 16px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(255,255,255,0.12);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #6B7280;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-size: 28px;
        font-weight: 800;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.7rem 1.4rem;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        letter-spacing: 0.01em;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }

    /* ── Select boxes ── */
    .stSelectbox > div > div {
        background: #141824;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        color: white;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #141824;
        border-radius: 10px;
        color: #6B7280;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: white !important;
        border-color: transparent;
    }

    /* ── Custom Components ── */
    .glass-card {
        background: linear-gradient(145deg, rgba(20,24,36,0.9), rgba(26,31,50,0.9));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 24px;
        margin: 10px 0;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .glass-card:hover {
        border-color: rgba(255,255,255,0.12);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }

    .gradient-text {
        background: linear-gradient(135deg, #6366F1, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    .accent-line {
        width: 40px;
        height: 3px;
        background: linear-gradient(90deg, #6366F1, #EC4899);
        border-radius: 2px;
        margin-bottom: 12px;
    }

    .divider {
        border: none;
        height: 1px;
        background: rgba(255,255,255,0.06);
        margin: 32px 0;
    }

    .match-result-card {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(20,24,36,0.8);
        border-radius: 12px;
        padding: 12px 18px;
        margin: 6px 0;
        border-left: 3px solid transparent;
        transition: all 0.2s ease;
    }
    .match-result-card:hover {
        background: rgba(26,31,50,0.9);
    }

    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-win {
        background: rgba(16,185,129,0.15);
        color: #10B981;
        border: 1px solid rgba(16,185,129,0.3);
    }
    .badge-loss {
        background: rgba(239,68,68,0.15);
        color: #EF4444;
        border: 1px solid rgba(239,68,68,0.3);
    }

    .winner-badge-large {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 10px 28px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
        margin-top: 16px;
        box-shadow: 0 4px 15px rgba(16,185,129,0.3);
    }

    .stat-label {
        color: #6B7280;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    .stat-value {
        color: #FFFFFF;
        font-size: 20px;
        font-weight: 700;
    }

    .info-box {
        background: rgba(99,102,241,0.08);
        border-left: 3px solid #6366F1;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        color: #B0B8C8;
        margin: 16px 0;
        font-size: 14px;
    }

    /* ── Justification Card ── */
    .justification-card {
        background: linear-gradient(145deg, #141824, #1A1F32);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }
    .factor-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .factor-row:last-child { border-bottom: none; }
    .factor-icon {
        width: 32px; height: 32px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; font-weight: 700;
    }
    .factor-positive { background: rgba(16,185,129,0.15); color: #10B981; }
    .factor-negative { background: rgba(239,68,68,0.15); color: #EF4444; }
    .factor-text { color: #E0E4EC; font-size: 14px; flex: 1; }
    .factor-value { color: #6B7280; font-size: 12px; font-weight: 600; min-width: 60px; text-align: right; }

    /* ── Feedback Stars ── */
    .star-rating { font-size: 28px; cursor: pointer; }
    .star-active { color: #F59E0B; }
    .star-inactive { color: #374151; }

    /* ── Gamification ── */
    .badge-earned {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 13px;
        color: #A5B4FC;
        margin: 4px;
    }

    .points-display {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 12px 20px;
        text-align: center;
    }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in { animation: fadeInUp 0.4s ease-out forwards; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════

@st.cache_data(ttl=300)
def load_data():
    """Load all required data files with caching."""
    data = {}

    matches_path = os.path.join("data", "processed", "matches.csv")
    if os.path.exists(matches_path):
        data["matches"] = pd.read_csv(matches_path, parse_dates=["date"])
    else:
        data["matches"] = None

    metadata_path = os.path.join("models", "training_metadata.pkl")
    if os.path.exists(metadata_path):
        import joblib
        data["metadata"] = joblib.load(metadata_path)
    else:
        data["metadata"] = None

    for name, filename in [("batting", "player_batting_stats.csv"), ("bowling", "player_bowling_stats.csv")]:
        path = os.path.join("data", "processed", filename)
        data[name] = pd.read_csv(path) if os.path.exists(path) else None

    backtest_path = os.path.join("data", "processed", "backtest_results.csv")
    data["backtest"] = pd.read_csv(backtest_path) if os.path.exists(backtest_path) else None

    return data


# ══════════════════════════════════════════
# PLOTLY THEME HELPER
# ══════════════════════════════════════════

def apply_dark_theme(fig, height=350):
    """Apply consistent dark theme to any Plotly figure."""
    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B0B8C8", family="Inter"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(font=dict(color="#B0B8C8")),
    )
    return fig


# ══════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════

def main():
    # Session state for anonymous user
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())[:8]

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0;">
            <div style="font-size: 40px; margin-bottom: 8px;">🏏</div>
            <div style="font-size: 18px; font-weight: 800; color: #FFF; letter-spacing: -0.02em;">IPL Predictor</div>
            <div style="font-size: 12px; color: #6B7280; margin-top: 4px;">AI-Powered Analysis</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["Dashboard", "Match Predictor", "Tournament", "Teams", "Players", "Model", "Community"],
            label_visibility="collapsed",
            format_func=lambda x: {
                "Dashboard": "🏠  Dashboard",
                "Match Predictor": "🔮  Predictor",
                "Tournament": "🏆  Tournament",
                "Teams": "📊  Teams",
                "Players": "⭐  Players",
                "Model": "📈  Model",
                "Community": "👥  Community",
            }.get(x, x),
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # User points display
        try:
            from src.dashboard.feedback import FeedbackManager
            fm = FeedbackManager()
            user_stats = fm.get_user_stats(st.session_state["session_id"])
            if user_stats["points"] > 0:
                st.markdown(f"""
                <div class="points-display">
                    <div style="font-size:11px; color:#6B7280; text-transform:uppercase; letter-spacing:0.1em;">Your Points</div>
                    <div style="font-size:24px; font-weight:800; color:#A5B4FC;">{user_stats['points']}</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass

        st.markdown("""
        <div style="text-align:center; padding: 20px 10px;">
            <p style="color: #374151; font-size: 11px;">
                Built by Goutham<br/>
                v2.0 | March 2026
            </p>
        </div>
        """, unsafe_allow_html=True)

    data = load_data()

    pages = {
        "Dashboard": show_dashboard,
        "Match Predictor": show_match_predictor,
        "Tournament": show_tournament_rankings,
        "Teams": show_team_analysis,
        "Players": show_player_stats,
        "Model": show_model_performance,
        "Community": show_community,
    }
    pages.get(page, show_dashboard)(data)


# ══════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════

def show_dashboard(data):
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("# IPL Prediction Agent")
    st.markdown('<p style="color: #6B7280; margin-top: -8px;">Machine learning meets cricket intelligence</p>', unsafe_allow_html=True)

    matches = data.get("matches")
    metadata = data.get("metadata")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matches", f"{len(matches):,}" if matches is not None else "0")
    with col2:
        acc = metadata.get("ensemble_accuracy", 0) * 100 if metadata else 0
        st.metric("Accuracy", f"{acc:.1f}%")
    with col3:
        st.metric("Features", f"{metadata.get('feature_count', 0)}" if metadata else "0")
    with col4:
        seasons = matches["season"].nunique() if matches is not None else 0
        st.metric("Seasons", f"{seasons}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if matches is not None and len(matches) > 0:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("### Win Leaderboard")
            wins = matches[
                matches["winner"].notna() & ~matches["winner"].isin(["no result", "tie"])
            ]["winner"].value_counts().head(10)
            colors = [TEAM_COLORS.get(t, "#555") for t in wins.index]

            fig = go.Figure(go.Bar(
                x=wins.values, y=[get_short_code(t) for t in wins.index],
                orientation="h", marker_color=colors,
                text=wins.values, textposition="outside",
                textfont=dict(color="white", size=12),
            ))
            apply_dark_theme(fig, 420)
            fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Wins")
            fig.update_layout(margin=dict(l=10, r=50, t=10, b=30))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### Recent Matches")
            recent = matches.sort_values("date", ascending=False).head(10)
            for _, m in recent.iterrows():
                t1 = get_short_code(m.get("team1", ""))
                t2 = get_short_code(m.get("team2", ""))
                winner = m.get("winner", "")
                w_short = get_short_code(winner) if winner else ""
                c1 = get_team_color(m.get("team1", ""))
                c2 = get_team_color(m.get("team2", ""))
                border_c = c1 if winner == m.get("team1") else c2

                t1_style = "font-weight:800;" if winner == m.get("team1") else "opacity:0.5;"
                t2_style = "font-weight:800;" if winner == m.get("team2") else "opacity:0.5;"

                st.markdown(f"""
                <div class="match-result-card" style="border-left-color:{border_c};">
                    <span style="color:{c1}; {t1_style} font-size:14px;">{t1}</span>
                    <span style="color:#374151; font-size:11px;">vs</span>
                    <span style="color:{c2}; {t2_style} font-size:14px;">{t2}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown("### Season Trend")
        sc = matches.groupby("season").size().reset_index(name="matches")
        fig2 = px.area(sc, x="season", y="matches", color_discrete_sequence=["#6366F1"])
        apply_dark_theme(fig2, 220)
        fig2.update_traces(line=dict(width=2), fillcolor="rgba(99,102,241,0.1)")
        fig2.update_layout(xaxis_title="", yaxis_title="Matches")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>Getting Started</strong><br/>
            No data found. Run the pipeline: <code>python run_pipeline.py</code>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: MATCH PREDICTOR (with justification + feedback)
# ══════════════════════════════════════════

def show_match_predictor(data):
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("# Match Predictor")
    st.markdown('<p style="color: #6B7280;">Select teams and get AI predictions with explanations</p>', unsafe_allow_html=True)

    teams = ACTIVE_TEAMS

    col1, _, col2 = st.columns([5, 1, 5])
    with col1:
        team1 = st.selectbox("Team 1", teams, index=0)
    with col2:
        team2 = st.selectbox("Team 2", [t for t in teams if t != team1], index=0)

    col3, col4 = st.columns(2)
    with col3:
        toss_winner = st.selectbox("Toss Winner", ["Unknown", team1, team2])
    with col4:
        toss_decision = st.selectbox("Toss Decision", ["Unknown", "Bat First", "Field First"])

    st.markdown("")
    predict_clicked = st.button("Predict Winner", type="primary", use_container_width=True)

    if predict_clicked:
        try:
            from src.models.predict import predict_match

            result = predict_match(
                team1, team2,
                toss_winner=toss_winner if toss_winner != "Unknown" else None,
                toss_decision=toss_decision.split()[0].lower() if toss_decision != "Unknown" else None,
            )

            if "error" not in result:
                prob1 = result["team1_win_probability"]
                prob2 = result["team2_win_probability"]
                winner = result["predicted_winner"]
                confidence = result["confidence"]
                c1 = get_team_color(team1)
                c2 = get_team_color(team2)
                s1 = get_short_code(team1)
                s2 = get_short_code(team2)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                # ── Prediction Display ──
                st.markdown(f"""
                <div class="glass-card animate-in" style="text-align:center;">
                    <div style="display:flex; justify-content:center; align-items:center; gap:50px; flex-wrap:wrap;">
                        <div>
                            <div style="font-size:13px; color:{c1}; text-transform:uppercase; letter-spacing:3px; font-weight:600;">{s1}</div>
                            <div style="font-size:52px; font-weight:900; color:{c1}; line-height:1.1;">{prob1:.0%}</div>
                            <div style="color:#6B7280; font-size:12px; margin-top:4px;">{team1}</div>
                        </div>
                        <div style="color:#374151; font-size:20px; font-weight:300;">VS</div>
                        <div>
                            <div style="font-size:13px; color:{c2}; text-transform:uppercase; letter-spacing:3px; font-weight:600;">{s2}</div>
                            <div style="font-size:52px; font-weight:900; color:{c2}; line-height:1.1;">{prob2:.0%}</div>
                            <div style="color:#6B7280; font-size:12px; margin-top:4px;">{team2}</div>
                        </div>
                    </div>
                    <div style="margin-top:24px;">
                        <span class="winner-badge-large">{get_short_code(winner)} wins — {confidence:.0f}% confidence</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── SHAP Justification ──
                top_factors = result.get("top_factors", [])
                if top_factors:
                    st.markdown("### Why this prediction?")
                    st.markdown('<div class="justification-card">', unsafe_allow_html=True)

                    for factor in top_factors[:6]:
                        is_positive = factor["direction"] == "favours_team1"
                        icon_class = "factor-positive" if is_positive else "factor-negative"
                        icon_text = "+" if is_positive else "−"
                        impact = abs(factor["shap_value"])
                        bar_width = min(100, impact * 400)

                        st.markdown(f"""
                        <div class="factor-row">
                            <div class="factor-icon {icon_class}">{icon_text}</div>
                            <div class="factor-text">{factor['reason']}</div>
                            <div class="factor-value">
                                <div style="width:{bar_width}px; height:4px; border-radius:2px;
                                     background:{'#10B981' if is_positive else '#EF4444'}; opacity:0.6;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # SHAP waterfall chart
                    chart_data = result.get("chart_data")
                    if chart_data and chart_data.get("labels"):
                        fig = go.Figure(go.Bar(
                            x=chart_data["values"],
                            y=chart_data["labels"],
                            orientation="h",
                            marker_color=chart_data["colors"],
                            text=[f"{v:+.3f}" for v in chart_data["values"]],
                            textposition="outside",
                            textfont=dict(color="#B0B8C8", size=11),
                        ))
                        apply_dark_theme(fig, 300)
                        fig.update_layout(
                            title="Feature Impact (SHAP Values)",
                            yaxis=dict(autorange="reversed"),
                            margin=dict(l=10, r=80, t=40, b=20),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                elif result.get("justification"):
                    st.markdown("### Why this prediction?")
                    st.markdown(f'<div class="info-box">{result["justification"]}</div>', unsafe_allow_html=True)

                # ── User Feedback Section ──
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown("### Your Prediction")
                st.markdown('<p style="color:#6B7280; font-size:13px;">Agree with the AI? Make your own call and earn points!</p>', unsafe_allow_html=True)

                fc1, fc2 = st.columns(2)
                with fc1:
                    user_pick = st.radio(
                        "Who do you think wins?",
                        [team1, team2],
                        horizontal=True,
                        key="user_prediction",
                    )
                with fc2:
                    user_reasoning = st.text_input(
                        "Your reasoning (optional — earns bonus points!)",
                        placeholder="e.g. Bumrah will dominate...",
                        key="user_reason",
                    )

                if st.button("Submit My Prediction", key="submit_feedback"):
                    try:
                        from src.dashboard.feedback import FeedbackManager
                        fm = FeedbackManager()
                        fb_result = fm.submit_prediction_feedback(
                            team1=team1, team2=team2,
                            ai_prediction=winner, user_prediction=user_pick,
                            user_reasoning=user_reasoning,
                            session_id=st.session_state["session_id"],
                        )
                        points = fb_result["points_earned"]
                        total = fb_result["total_points"]

                        st.success(f"+{points} points! (Total: {total})")

                        new_badges = fb_result.get("badges", [])
                        if new_badges:
                            badge_html = " ".join([
                                f'<span class="badge-earned">{b["icon"]} {b["name"]}</span>'
                                for b in new_badges
                            ])
                            st.markdown(f"<div>{badge_html}</div>", unsafe_allow_html=True)
                    except Exception:
                        st.info("Prediction noted!")

                # Justification rating
                st.markdown("")
                st.markdown('<p style="color:#6B7280; font-size:13px;">How helpful was the AI\'s explanation?</p>', unsafe_allow_html=True)
                rating = st.slider("Rate 1-5", 1, 5, 3, key="justification_rating", label_visibility="collapsed")
                if st.button("Submit Rating", key="submit_rating"):
                    try:
                        from src.dashboard.feedback import FeedbackManager
                        fm = FeedbackManager()
                        fm.submit_justification_rating(
                            team1=team1, team2=team2, rating=rating,
                            session_id=st.session_state["session_id"],
                        )
                        st.success(f"Thanks! +5 points for rating.")
                    except Exception:
                        st.info("Rating noted!")

            else:
                st.error(result["error"])

        except Exception as e:
            st.markdown(f"""
            <div class="info-box">
                <strong>Model not trained yet</strong><br/>
                Run: <code>python run_pipeline.py</code><br/>
                <small style="color:#4B5563;">{e}</small>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: TOURNAMENT RANKINGS
# ══════════════════════════════════════════

def show_tournament_rankings(data):
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("# Tournament Predictions")
    st.markdown('<p style="color: #6B7280;">AI-ranked probability of winning IPL 2026</p>', unsafe_allow_html=True)

    if st.button("Generate Rankings", type="primary"):
        try:
            from src.models.predict import predict_tournament_winner
            with st.spinner("Simulating all matchups..."):
                rankings = predict_tournament_winner()

            if rankings:
                for r in rankings:
                    team = r["team"]
                    prob = r["win_probability"]
                    rank = r["rank"]
                    color = get_team_color(team)
                    short = get_short_code(team)

                    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"<span style='color:#6B7280;font-weight:600;'>#{rank}</span>")
                    bar_pct = prob * 100

                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:14px;
                                background:rgba(20,24,36,0.8); border-radius:12px; padding:14px 20px; margin:6px 0;
                                border-left: 3px solid {color};">
                        <span style="font-size:18px; min-width:32px; text-align:center;">{medal}</span>
                        <span style="color:{color}; font-weight:700; font-size:15px; min-width:48px;">{short}</span>
                        <div style="flex:1; background:rgba(255,255,255,0.04); border-radius:6px; height:20px; overflow:hidden;">
                            <div style="width:{bar_pct}%; height:100%; background:linear-gradient(90deg, {color}, {color}66);
                                        border-radius:6px;"></div>
                        </div>
                        <span style="color:#FFF; font-weight:700; min-width:55px; text-align:right;">{prob:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="info-box">Train the model first: <code>python run_pipeline.py</code></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">Click <strong>Generate Rankings</strong> to simulate matchups.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: TEAM ANALYSIS
# ══════════════════════════════════════════

def show_team_analysis(data):
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("# Team Analysis")

    matches = data.get("matches")
    if matches is None:
        st.markdown('<div class="info-box">Run <code>python run_pipeline.py</code> first.</div>', unsafe_allow_html=True)
        return

    all_teams = sorted(set(matches["team1"].unique()) | set(matches["team2"].unique()))
    active = [t for t in all_teams if t in TEAM_COLORS]
    selected = st.selectbox("Select Team", active if active else all_teams)
    tc = get_team_color(selected)

    tm = matches[(matches["team1"] == selected) | (matches["team2"] == selected)].sort_values("date")
    tm["won"] = (tm["winner"] == selected).astype(int)
    total, wins = len(tm), int(tm["won"].sum())

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Matches", total)
    with col2: st.metric("Wins", wins)
    with col3: st.metric("Losses", total - wins)
    with col4: st.metric("Win Rate", f"{wins/max(total,1):.1%}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    tm["rolling_wr"] = tm["won"].rolling(15, min_periods=3).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tm["date"], y=tm["rolling_wr"],
        mode="lines", fill="tozeroy",
        line=dict(color=tc, width=2),
        fillcolor=f"rgba({int(tc[1:3],16)},{int(tc[3:5],16)},{int(tc[5:7],16)},0.1)",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.1)")
    apply_dark_theme(fig, 320)
    fig.update_layout(title=f"Rolling 15-Match Win Rate", yaxis=dict(range=[0, 1], title="Win Rate"))
    st.plotly_chart(fig, use_container_width=True)

    ss = tm.groupby("season").agg(played=("won", "count"), wins=("won", "sum")).reset_index()
    ss["wr"] = ss["wins"] / ss["played"]
    fig2 = go.Figure(go.Bar(
        x=ss["season"], y=ss["wr"],
        marker_color=[tc if wr >= 0.5 else "#EF4444" for wr in ss["wr"]],
        text=[f"{wr:.0%}" for wr in ss["wr"]],
        textposition="outside", textfont=dict(color="white", size=11),
    ))
    fig2.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.1)")
    apply_dark_theme(fig2, 280)
    fig2.update_layout(title="Win Rate by Season", yaxis=dict(range=[0, 1], title="Win Rate"))
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════
# PAGE: PLAYER STATS
# ══════════════════════════════════════════

def show_player_stats(data):
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("# Player Statistics")

    batting, bowling = data.get("batting"), data.get("bowling")
    if batting is None and bowling is None:
        st.markdown('<div class="info-box">Run <code>python -m src.data_collection.build_player_stats</code> first.</div>', unsafe_allow_html=True)
        return

    tab1, tab2 = st.tabs(["Batting", "Bowling"])

    with tab1:
        if batting is not None:
            min_m = st.slider("Min matches", 5, 100, 20, key="bm")
            f = batting[batting["matches"] >= min_m].nlargest(20, "total_runs")
            fig = go.Figure(go.Bar(
                x=f["total_runs"], y=f["batter"], orientation="h",
                marker_color="#6366F1",
                text=[f"{r:,} (SR {sr:.0f})" for r, sr in zip(f["total_runs"], f["strike_rate"])],
                textposition="outside", textfont=dict(color="#B0B8C8", size=11),
            ))
            apply_dark_theme(fig, max(400, len(f) * 28))
            fig.update_layout(yaxis=dict(autorange="reversed"), title="Top Run Scorers", margin=dict(r=120))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if bowling is not None:
            min_m2 = st.slider("Min matches", 5, 100, 20, key="bwm")
            fb = bowling[bowling["matches"] >= min_m2].nlargest(20, "wickets")
            fig2 = go.Figure(go.Bar(
                x=fb["wickets"], y=fb["bowler"], orientation="h",
                marker_color="#EC4899",
                text=[f"{w:.0f} wkts (Econ {e:.1f})" for w, e in zip(fb["wickets"], fb["economy"])],
                textposition="outside", textfont=dict(color="#B0B8C8", size=11),
            ))
            apply_dark_theme(fig2, max(400, len(fb) * 28))
            fig2.update_layout(yaxis=dict(autorange="reversed"), title="Top Wicket Takers", margin=dict(r=130))
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════

def show_model_performance(data):
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("# Model Performance")

    metadata = data.get("metadata")
    backtest = data.get("backtest")

    if metadata:
        models = {
            "XGBoost": metadata.get("xgboost_accuracy", 0),
            "LightGBM": metadata.get("lightgbm_accuracy", 0),
            "Logistic": metadata.get("logistic_accuracy", 0),
            "Ensemble": metadata.get("ensemble_accuracy", 0),
        }
        colors = ["#6366F1", "#8B5CF6", "#6B7280", "#10B981"]

        fig = go.Figure(go.Bar(
            x=list(models.keys()), y=[v * 100 for v in models.values()],
            marker_color=colors,
            text=[f"{v*100:.1f}%" for v in models.values()],
            textposition="outside", textfont=dict(color="white", size=15, family="Inter"),
        ))
        fig.add_hline(y=65, line_dash="dash", line_color="#EF4444",
                      annotation_text="Target: 65%", annotation_font_color="#EF4444")
        apply_dark_theme(fig, 350)
        fig.update_layout(yaxis=dict(range=[0, 100], title="Accuracy %"))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Training Samples", f"{metadata.get('training_samples', 0):,}")
        with col2: st.metric("Test Samples", f"{metadata.get('test_samples', 0):,}")
        with col3: st.metric("Trained", metadata.get("trained_at", "N/A")[:10])

    if backtest is not None:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### Back-Test Results")
        fig2 = go.Figure()
        for adaptive, name, color in [(False, "Static", "#6B7280"), (True, "Adaptive", "#10B981")]:
            s = backtest[backtest["adaptive"] == adaptive]
            fig2.add_trace(go.Bar(
                x=s["season"], y=s["accuracy"] * 100, name=name, marker_color=color,
                text=[f"{a*100:.0f}%" for a in s["accuracy"]],
                textposition="outside", textfont=dict(color="white"),
            ))
        apply_dark_theme(fig2, 350)
        fig2.update_layout(barmode="group", yaxis_title="Accuracy %")
        st.plotly_chart(fig2, use_container_width=True)

    if not metadata and backtest is None:
        st.markdown('<div class="info-box">Train the model: <code>python -m src.models.train_model</code></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════
# PAGE: COMMUNITY (Feedback + Leaderboard)
# ══════════════════════════════════════════

def show_community(data):
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("# Community")
    st.markdown('<p style="color: #6B7280;">See how the crowd compares against the AI</p>', unsafe_allow_html=True)

    try:
        from src.dashboard.feedback import FeedbackManager
        fm = FeedbackManager()
        community = fm.get_community_stats()
        user_stats = fm.get_user_stats(st.session_state["session_id"])

        # User stats
        st.markdown("### Your Stats")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Points", user_stats["points"])
        with col2: st.metric("Predictions", user_stats["predictions"])
        with col3: st.metric("Accuracy", f"{user_stats['accuracy']}%")
        with col4: st.metric("Best Streak", user_stats["best_streak"])

        # Badges
        badges = user_stats.get("badges", [])
        if badges:
            badge_html = " ".join([
                f'<span class="badge-earned">{b["icon"]} {b["name"]}</span>'
                for b in badges
            ])
            st.markdown(f"<div style='margin:12px 0;'>{badge_html}</div>", unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Community stats
        st.markdown("### Community Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Predictions", community["total_predictions"])
        with c2: st.metric("AI Accuracy", f"{community['ai_accuracy']}%")
        with c3: st.metric("Crowd Accuracy", f"{community['crowd_accuracy']}%")
        with c4: st.metric("Avg Rating", f"{community['avg_rating']}/5")

        # Leaderboard
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### Leaderboard")
        lb = fm.get_leaderboard(10)
        if lb:
            for entry in lb:
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(entry["rank"], f"#{entry['rank']}")
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:14px;
                            background:rgba(20,24,36,0.8); border-radius:10px; padding:10px 18px; margin:5px 0;">
                    <span style="font-size:16px; min-width:30px;">{medal}</span>
                    <span style="color:#A5B4FC; font-weight:600; min-width:80px;">{entry['session_id']}</span>
                    <span style="color:#FFF; font-weight:700; flex:1;">{entry['points']} pts</span>
                    <span style="color:#6B7280; font-size:13px;">{entry['predictions']} picks · {entry['accuracy']}% acc</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">No predictions yet. Be the first!</div>', unsafe_allow_html=True)

    except Exception:
        st.markdown("""
        <div class="info-box">
            Community features will be available after making your first prediction on the Match Predictor page.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
