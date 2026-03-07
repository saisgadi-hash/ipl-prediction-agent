"""
IPL Prediction Agent — Streamlit Dashboard v2.0
Apple Design Award-level UI with SHAP justifications and user feedback.

HOW TO RUN:
    streamlit run src/dashboard/app.py
"""

import os
import sys
import uuid
import base64
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "data_collection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "models"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "dashboard"))
from team_name_mapper import (
    standardise_team_name, get_short_code, ACTIVE_TEAMS, TEAM_SHORT_CODES,
)
import __main__
from train_model import IPLEnsemblePredictor
__main__.IPLEnsemblePredictor = IPLEnsemblePredictor

# ── Page Config ──
st.set_page_config(
    page_title="IPL Prediction Agent",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Team Colors (official IPL team colors) ──
TEAM_COLORS = {
    "Chennai Super Kings": "#F5A623",
    "Mumbai Indians": "#0984E3",
    "Royal Challengers Bengaluru": "#E74C3C",
    "Kolkata Knight Riders": "#8E44AD",
    "Delhi Capitals": "#2980B9",
    "Punjab Kings": "#E74C3C",
    "Rajasthan Royals": "#E84393",
    "Sunrisers Hyderabad": "#F39C12",
    "Lucknow Super Giants": "#00CEC9",
    "Gujarat Titans": "#636E72",
    "Rising Pune Supergiant": "#8E44AD",
    "Pune Warriors": "#2F9BE3",
    "Deccan Chargers": "#95A5A6",
    "Kochi Tuskers Kerala": "#8E44AD",
    "Gujarat Lions": "#E67E22",
}

# Light tints for card backgrounds
TEAM_COLORS_LIGHT = {
    "Chennai Super Kings": "#FEF3E0",
    "Mumbai Indians": "#E3F2FD",
    "Royal Challengers Bengaluru": "#FFEBEE",
    "Kolkata Knight Riders": "#F3E5F5",
    "Delhi Capitals": "#E1F5FE",
    "Punjab Kings": "#FFEBEE",
    "Rajasthan Royals": "#FCE4EC",
    "Sunrisers Hyderabad": "#FFF8E1",
    "Lucknow Super Giants": "#E0F7FA",
    "Gujarat Titans": "#ECEFF1",
}


def get_team_color(name: str) -> str:
    return TEAM_COLORS.get(standardise_team_name(name), "#888888")


TEAM_SHORT = TEAM_SHORT_CODES


# ══════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════

@st.cache_data
def get_image_base64(img_path):
    try:
        with open(img_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
            return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


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
        background: linear-gradient(135deg, #FAFBFE 0%, #F0F4FF 50%, #FFF5F5 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid rgba(108, 92, 231, 0.08);
        box-shadow: 2px 0 20px rgba(108, 92, 231, 0.04);
    }
    section[data-testid="stSidebar"] * { color: #4A5568 !important; }

    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6 {
        color: #2D3748 !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    h1 { font-size: 2.2rem !important; }
    p, span, div, label, li { color: #4A5568; }
    .stMarkdown p, .stMarkdown span, .stText p { color: #2D3748 !important; }
    .stRadio label, .stSelectbox label, .stTextInput label, .stTextArea label, .stNumberInput label {
        color: #2D3748 !important; font-weight: 600 !important;
    }

    /* ── Metric Cards (each gets a unique colour via inline style) ── */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: none;
        border-radius: 16px;
        padding: 22px 18px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);
        transition: all 0.25s ease;
        border-top: 3px solid #6C5CE7;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 8px 25px rgba(108, 92, 231, 0.12);
    }
    div[data-testid="stMetric"] label {
        color: #718096; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.1em; font-weight: 700;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #2D3748; font-size: 28px; font-weight: 800;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6C5CE7, #A29BFE);
        color: white; border: none; border-radius: 12px;
        font-weight: 600; padding: 0.7rem 1.6rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.25);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.35);
        background: linear-gradient(135deg, #5B4BD5, #8B7FF0);
    }

    /* ── Select boxes ── */
    .stSelectbox > div > div {
        background: #FFFFFF; border: 2px solid #EDF2F7;
        border-radius: 12px; color: #2D3748;
        transition: border-color 0.2s ease;
    }
    .stSelectbox > div > div:focus-within { border-color: #6C5CE7; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: #FFFFFF; border-radius: 10px; color: #718096;
        border: 1px solid #EDF2F7; padding: 8px 18px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6C5CE7, #A29BFE);
        color: white !important; border-color: transparent;
        box-shadow: 0 3px 12px rgba(108, 92, 231, 0.2);
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 92, 231, 0.06);
        border-radius: 18px; padding: 24px; margin: 10px 0;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
        transition: all 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(108, 92, 231, 0.08);
    }

    /* ── Gradient Text ── */
    .gradient-text {
        background: linear-gradient(135deg, #6C5CE7, #00B894, #FDCB6E, #E17055);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* ── Accent Line ── */
    .accent-line {
        width: 50px; height: 4px;
        background: linear-gradient(90deg, #6C5CE7, #00B894, #FDCB6E, #E17055);
        border-radius: 2px; margin-bottom: 16px;
    }

    .divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, #E2E8F0, transparent); margin: 32px 0; }

    /* ── Match Result Card ── */
    .match-result-card {
        display: flex; justify-content: space-between; align-items: center;
        background: #FFFFFF; border-radius: 12px; padding: 14px 20px; margin: 6px 0;
        border-left: 4px solid transparent;
        box-shadow: 0 1px 6px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .match-result-card:hover { background: #FAFBFE; transform: translateX(4px); }

    /* ── Badges ── */
    .badge { display: inline-block; padding: 5px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; letter-spacing: 0.03em; }
    .badge-win { background: linear-gradient(135deg, rgba(0,184,148,0.1), rgba(0,184,148,0.05)); color: #00B894; border: 1px solid rgba(0,184,148,0.2); }
    .badge-loss { background: linear-gradient(135deg, rgba(225,112,85,0.1), rgba(225,112,85,0.05)); color: #E17055; border: 1px solid rgba(225,112,85,0.2); }

    .winner-badge-large {
        background: linear-gradient(135deg, #00B894, #55EFC4);
        color: white; padding: 12px 32px; border-radius: 28px;
        font-weight: 700; font-size: 16px; display: inline-block; margin-top: 16px;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
    }

    /* ── Info Box ── */
    .info-box {
        background: linear-gradient(135deg, #EBF5FF, #F0E6FF);
        border-left: 3px solid #6C5CE7;
        padding: 16px 20px; border-radius: 0 12px 12px 0;
        color: #2D3748; margin: 16px 0; font-size: 14px;
    }

    /* ── Justification Card ── */
    .justification-card {
        background: #FFFFFF; border: 1px solid #EDF2F7;
        border-radius: 16px; padding: 24px; margin: 16px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    .factor-row { display: flex; align-items: center; gap: 12px; padding: 12px 0; border-bottom: 1px solid #F7FAFC; }
    .factor-row:last-child { border-bottom: none; }
    .factor-icon { width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700; }
    .factor-positive { background: rgba(0, 184, 148, 0.1); color: #00B894; }
    .factor-negative { background: rgba(225, 112, 85, 0.1); color: #E17055; }
    .factor-text { color: #2D3748; font-size: 14px; flex: 1; }
    .factor-value { color: #718096; font-size: 12px; font-weight: 700; min-width: 60px; text-align: right; }

    /* ── Gamification ── */
    .badge-earned {
        display: inline-flex; align-items: center; gap: 6px;
        background: linear-gradient(135deg, rgba(108,92,231,0.08), rgba(162,155,254,0.05));
        border: 1px solid rgba(108,92,231,0.15); border-radius: 22px;
        padding: 7px 16px; font-size: 13px; color: #6C5CE7; margin: 4px; font-weight: 600;
    }

    .points-display {
        background: linear-gradient(135deg, #6C5CE7, #A29BFE);
        border: none; border-radius: 14px; padding: 14px 20px;
        text-align: center; color: white !important;
        box-shadow: 0 4px 15px rgba(108,92,231,0.2);
    }
    .points-display * { color: white !important; }

    /* ── Animations ── */
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .animate-in { animation: fadeInUp 0.4s ease-out forwards; }
    @keyframes shimmer { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }
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

def apply_light_theme(fig, height=350):
    """Apply consistent clean & colourful theme to any Plotly figure."""
    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2D3748", family="Inter", size=13),
        xaxis=dict(gridcolor="#F0F4FF", zerolinecolor="#E2E8F0", tickfont=dict(color="#718096", size=11)),
        yaxis=dict(gridcolor="#F0F4FF", zerolinecolor="#E2E8F0", tickfont=dict(color="#718096", size=11)),
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(font=dict(color="#2D3748", size=12)),
    )
    return fig


# Vibrant colour palette for charts
CHART_COLORS = ["#6C5CE7", "#00B894", "#FDCB6E", "#E17055", "#74B9FF",
                "#A29BFE", "#55EFC4", "#FFEAA7", "#FAB1A0", "#81ECEC",
                "#FF7675", "#FD79A8", "#0984E3", "#00CEC9", "#E84393"]


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
            <div style="font-size: 18px; font-weight: 800; color: #1C1E21; letter-spacing: -0.02em;">IPL Predictor</div>
            <div style="font-size: 12px; color: #6B7280; margin-top: 4px;">AI-Powered Analysis</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Initialize navigation state if not present
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "Dashboard"

        nav_items = [
            ("Dashboard", "nav_dashboard_1772835882132.png"),
            ("Live Match", "nav_dashboard_1772835882132.png"),
            ("Match Predictor", "nav_predictor_1772835896572.png"),
            ("Tournament", "nav_tournament_1772835910419.png"),
            ("Teams", "nav_teams_1772835924417.png"),
            ("Players", "nav_players_1772836000895.png"),
            ("Model", "nav_model_1772835977122.png"),
            ("Community", "nav_community_1772835957621.png"),
        ]

        st.markdown(
            """
            <style>
            .stButton.nav-button > button {
                width: 100%;
                text-align: left !important;
                background-color: transparent !important;
                color: #4B4F56 !important;
                border: none !important;
                padding: 10px 15px !important;
                border-radius: 8px !important;
                font-size: 15px !important;
                font-weight: 600 !important;
                display: flex !important;
                justify-content: flex-start !important;
                gap: 12px !important;
                align-items: center !important;
                box-shadow: none !important;
                transition: all 0.2s ease;
            }
            .stButton.nav-button > button:hover {
                background-color: #F0F2F5 !important;
                color: #1C1E21 !important;
            }
            .stButton.nav-button.active-nav > button {
                background-color: #EBF5FF !important;
                color: #1877F2 !important;
            }
            div[data-testid="stVerticalBlock"] > div > div > div > div > div.stButton {
                margin-bottom: 4px;
            }
            /* Hide the default Streamlit button markdown output p tags to let flex work */
            .stButton.nav-button > button p {
                margin: 0 !important;
                flex: 1;
                text-align: left;
            }
            </style>
            """, unsafe_allow_html=True
        )

        for label, icon_file in nav_items:
            icon_path = Path("src/dashboard/assets") / icon_file
            img_b64 = get_image_base64(icon_path)
            
            # Using HTML inside the button label to render the image alongside text
            button_html = f'<img src="{img_b64}" width="24" height="24" style="border-radius:6px;"> <span style="margin-top:2px">{label}</span>'
            
            # We use a container to apply the right class name depending on state
            if st.session_state["current_page"] == label:
                button_class = "nav-button active-nav"
            else:
                button_class = "nav-button"
                
            # Streamlit doesn't natively allow rich HTML in buttons, but markdown works with some tricks.
            # Instead of standard buttons, we'll use columns to layout image and button nicely.
            col_icon, col_btn = st.columns([1, 4])
            with col_icon:
                st.markdown(f'<img src="{img_b64}" width="28" height="28" style="margin-top: 5px; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
            with col_btn:
                if st.button(label, key=f"nav_{label}", use_container_width=True):
                    st.session_state["current_page"] = label
                    st.rerun()

        page = st.session_state["current_page"]

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # User points display
        try:
            from feedback import FeedbackManager
            fm = FeedbackManager()
            user_stats = fm.get_user_stats(st.session_state["session_id"])
            if user_stats["points"] > 0:
                st.markdown(f"""
                <div class="points-display">
                    <div style="font-size:11px; color:#6B7280; text-transform:uppercase; letter-spacing:0.1em;">Your Points</div>
                    <div style="font-size:24px; font-weight:800; color:#1877F2;">{user_stats['points']}</div>
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
        "Live Match": show_live_match,
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
                textfont=dict(color="#1C1E21", size=12),
            ))
            apply_light_theme(fig, 420)
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
        apply_light_theme(fig2, 220)
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
# PAGE: LIVE MATCH (Phase D)
# ══════════════════════════════════════════

def show_live_match(data):
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("# Live Match")
    st.markdown('<p style="color: #6B7280;">Real-time scores, predictions, and betting odds during live IPL matches</p>', unsafe_allow_html=True)

    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh every 30 seconds", value=False)
    if auto_refresh:
        import time as _t
        _t.sleep(0.1)
        st.markdown('<meta http-equiv="refresh" content="30">', unsafe_allow_html=True)

    try:
        from live_score_collector import LiveScoreCollector
        collector = LiveScoreCollector()
        matches = collector.get_live_matches()
    except Exception as e:
        matches = []
        st.markdown(f'<div class="info-box">Live score service unavailable: {e}. Add CRICAPI_KEY to your .env file.</div>', unsafe_allow_html=True)

    if not matches:
        # No live matches — show helpful message
        st.markdown("""
        <div style="text-align:center; padding:60px 20px;">
            <div style="font-size:48px; margin-bottom:16px;">📡</div>
            <h3 style="color:#2D3748;">No Live IPL Matches Right Now</h3>
            <p style="color:#6B7280; max-width:400px; margin:0 auto;">
                Live scores appear here during IPL matches. Check back when a match is in progress!
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Show upcoming odds if available
        try:
            from odds_collector import OddsCollector
            odds_collector = OddsCollector()
            odds = odds_collector.get_upcoming_odds()
            if odds:
                st.markdown("### Upcoming Match Odds")
                for o in odds:
                    t1 = o.get("team1", "TBD")
                    t2 = o.get("team2", "TBD")
                    p1 = o.get("implied_probability_team1", 0.5)
                    p2 = o.get("implied_probability_team2", 0.5)
                    st.markdown(f"""
                    <div style="background:white; border-radius:12px; padding:16px 20px; margin:8px 0;
                                border-left:3px solid #6C5CE7; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:700; color:#2D3748;">{t1}</span>
                            <span style="color:#6B7280; font-size:12px;">vs</span>
                            <span style="font-weight:700; color:#2D3748;">{t2}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:8px;">
                            <span style="color:#6C5CE7; font-weight:600;">{p1:.1%}</span>
                            <span style="color:#6B7280; font-size:11px;">Bookmaker Implied Probability</span>
                            <span style="color:#E17055; font-weight:600;">{p2:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            pass
        return

    # ── Live match selector ──
    match_names = [f"{m['team1']} vs {m['team2']} — {m['status']}" for m in matches]
    selected_idx = st.selectbox("Select Match", range(len(match_names)), format_func=lambda i: match_names[i])
    match = matches[selected_idx]

    team1 = match.get("team1", "")
    team2 = match.get("team2", "")
    t1_color = get_team_color(team1)
    t2_color = get_team_color(team2)

    # ── Live Scoreboard ──
    st.markdown("### Live Score")
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background:white; border-radius:12px;
                    border-top:3px solid {t1_color}; box-shadow:0 2px 8px rgba(0,0,0,0.06);">
            <div style="font-weight:700; color:{t1_color}; font-size:16px;">{get_short_code(team1)}</div>
            <div style="font-size:28px; font-weight:800; color:#2D3748; margin:8px 0;">
                {match.get('team1_score', '-')}
            </div>
            <div style="color:#6B7280; font-size:12px;">RR: {match.get('team1_run_rate', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align:center; padding:20px;">
            <div style="font-size:11px; color:#6B7280; text-transform:uppercase;">vs</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background:white; border-radius:12px;
                    border-top:3px solid {t2_color}; box-shadow:0 2px 8px rgba(0,0,0,0.06);">
            <div style="font-weight:700; color:{t2_color}; font-size:16px;">{get_short_code(team2)}</div>
            <div style="font-size:28px; font-weight:800; color:#2D3748; margin:8px 0;">
                {match.get('team2_score', '-') or '-'}
            </div>
            <div style="color:#6B7280; font-size:12px;">RR: {match.get('team2_run_rate', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Status bar
    st.markdown(f"""
    <div style="text-align:center; padding:10px; color:#6B7280; font-size:13px;">
        {match.get('status', '')} | {match.get('venue', '')}
    </div>
    """, unsafe_allow_html=True)

    # ── Live Prediction ──
    if match.get("match_started") and not match.get("match_ended"):
        st.markdown("### AI Prediction")
        try:
            from predict import predict_live
            prediction = predict_live(match)

            live_prob_t1 = prediction.get("live_probability_team1", 0.5)
            live_prob_t2 = prediction.get("live_probability_team2", 0.5)
            winner = prediction.get("predicted_winner", "")
            momentum = prediction.get("momentum", "neutral")
            projected = prediction.get("projected_score", 0)
            adjustment = prediction.get("adjustment", 0)

            momentum_icon = {
                "strong_positive": "🚀", "positive": "📈",
                "neutral": "➡️",
                "negative": "📉", "strong_negative": "💥",
            }.get(momentum, "➡️")

            # Win probability bar
            pct1 = int(live_prob_t1 * 100)
            pct2 = 100 - pct1
            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:20px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06); margin:10px 0;">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:{t1_color}; font-weight:700;">{get_short_code(team1)} {pct1}%</span>
                    <span style="font-size:14px;">{momentum_icon} {momentum.replace('_', ' ').title()}</span>
                    <span style="color:{t2_color}; font-weight:700;">{pct2}% {get_short_code(team2)}</span>
                </div>
                <div style="display:flex; height:24px; border-radius:12px; overflow:hidden;">
                    <div style="width:{pct1}%; background:{t1_color};"></div>
                    <div style="width:{pct2}%; background:{t2_color};"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:12px; font-size:12px; color:#6B7280;">
                    <span>Projected Score: {projected}</span>
                    <span>Adjustment: {adjustment:+.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="info-box">Live prediction unavailable: {e}</div>', unsafe_allow_html=True)

    # ── Betting Odds Comparison ──
    try:
        from odds_collector import OddsCollector
        odds_collector = OddsCollector()
        all_odds = odds_collector.get_upcoming_odds()

        match_odds = None
        for o in all_odds:
            if (team1 in o.get("team1", "") or team1 in o.get("team2", "")) and \
               (team2 in o.get("team1", "") or team2 in o.get("team2", "")):
                match_odds = o
                break

        if match_odds:
            st.markdown("### Model vs Bookmakers")
            imp_t1 = match_odds.get("implied_probability_team1", 0.5)
            imp_t2 = match_odds.get("implied_probability_team2", 0.5)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{get_short_code(team1)} (Bookmaker)", f"{imp_t1:.1%}")
            with col2:
                st.metric(f"{get_short_code(team2)} (Bookmaker)", f"{imp_t2:.1%}")

            # Show bookmaker details
            for bm_name, bm_data in match_odds.get("bookmakers", {}).items():
                odds_str = " | ".join([f"{k}: {v['decimal_odds']}" for k, v in bm_data.items()])
                st.markdown(f"<span style='color:#6B7280; font-size:12px;'>{bm_name}: {odds_str}</span>", unsafe_allow_html=True)

    except Exception:
        pass

    # ── Match Info ──
    st.markdown(f"""
    <div style="text-align:center; color:#6B7280; font-size:11px; margin-top:20px;">
        Last updated: {match.get('last_updated', 'N/A')} |
        API requests remaining today: {collector.get_requests_remaining() if 'collector' in dir() else 'N/A'}
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
            from predict import predict_match

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
                        apply_light_theme(fig, 300)
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
                        from feedback import FeedbackManager
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
                        from feedback import FeedbackManager
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
    st.markdown('<p style="color: #6B7280;">Monte Carlo simulation — 5,000 full season simulations with Elo-based probabilities</p>', unsafe_allow_html=True)

    if st.button("Run Monte Carlo Simulation", type="primary"):
        try:
            from predict import predict_tournament_winner
            with st.spinner("Running 5,000 season simulations..."):
                rankings = predict_tournament_winner()

            if rankings:
                # ── Monte Carlo Rankings ──
                st.markdown("### Win Probability Rankings")
                for r in rankings:
                    team = r["team"]
                    prob = r["win_probability"]
                    rank = r["rank"]
                    color = get_team_color(team)
                    light_color = TEAM_COLORS_LIGHT.get(team, "#F0F4FF")
                    short = get_short_code(team)
                    elo = r.get("elo_rating", 1500)
                    form = r.get("form_state", "Normal")
                    form_icon = {"Hot": "🔥", "Normal": "⚡", "Cold": "❄️"}.get(form, "⚡")

                    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"<span style='color:#6B7280;font-weight:600;'>#{rank}</span>")
                    bar_pct = prob * 100

                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:14px;
                                background:{light_color}; border-radius:12px; padding:14px 20px; margin:6px 0;
                                border-left: 3px solid {color};">
                        <span style="font-size:18px; min-width:32px; text-align:center;">{medal}</span>
                        <span style="color:{color}; font-weight:700; font-size:15px; min-width:48px;">{short}</span>
                        <div style="flex:1; background:rgba(0,0,0,0.06); border-radius:6px; height:20px; overflow:hidden;">
                            <div style="width:{bar_pct}%; height:100%; background:linear-gradient(90deg, {color}, {color}99);
                                        border-radius:6px;"></div>
                        </div>
                        <span style="color:#2D3748; font-weight:700; min-width:55px; text-align:right;">{prob:.1%}</span>
                        <span style="font-size:12px; color:#6B7280; min-width:70px; text-align:right;">Elo {elo:.0f}</span>
                        <span style="font-size:14px; min-width:24px;">{form_icon}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Bar Chart with 95% Confidence Intervals ──
                st.markdown("### Monte Carlo Distribution")
                import plotly.graph_objects as go
                fig = go.Figure()
                teams_sorted = [r["team"] for r in rankings]
                probs_sorted = [r["win_probability"] * 100 for r in rankings]
                colors_sorted = [get_team_color(t) for t in teams_sorted]

                # Phase E: error bars from CI
                ci_lowers = [r.get("ci_lower", r["win_probability"]) * 100 for r in rankings]
                ci_uppers = [r.get("ci_upper", r["win_probability"]) * 100 for r in rankings]
                err_minus = [p - lo for p, lo in zip(probs_sorted, ci_lowers)]
                err_plus = [hi - p for p, hi in zip(probs_sorted, ci_uppers)]

                fig.add_trace(go.Bar(
                    x=[get_short_code(t) for t in teams_sorted],
                    y=probs_sorted,
                    marker_color=colors_sorted,
                    text=[f"{p:.1f}%" for p in probs_sorted],
                    textposition="outside",
                    error_y=dict(
                        type="data", symmetric=False,
                        array=err_plus, arrayminus=err_minus,
                        color="rgba(0,0,0,0.3)", thickness=1.5, width=4,
                    ),
                    hovertemplate="<b>%{x}</b><br>Win: %{y:.1f}%<br>95% CI: %{customdata[0]:.1f}%–%{customdata[1]:.1f}%<extra></extra>",
                    customdata=list(zip(ci_lowers, ci_uppers)),
                ))
                apply_light_theme(fig)
                fig.update_layout(
                    title="Tournament Win Probability with 95% Confidence Interval",
                    yaxis_title="Win Probability (%)",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Phase E: Pythagorean Win Expectation ──
                has_pwe = any(r.get("pwe_expected", 0.5) != 0.5 for r in rankings)
                if has_pwe:
                    st.markdown("### Pythagorean Win Expectation")
                    st.markdown('<p style="color: #6B7280;">Expected win % from runs scored vs conceded (RS<sup>6.5</sup> / (RS<sup>6.5</sup> + RC<sup>6.5</sup>)). Teams above expected are overperforming.</p>', unsafe_allow_html=True)

                    pwe_fig = go.Figure()
                    pwe_teams = [get_short_code(r["team"]) for r in rankings]
                    pwe_expected = [r.get("pwe_expected", 0.5) * 100 for r in rankings]
                    pwe_actual = [r.get("pwe_actual", 0.5) * 100 for r in rankings]
                    pwe_colors = [get_team_color(r["team"]) for r in rankings]

                    pwe_fig.add_trace(go.Bar(
                        name="Expected (Pythagorean)",
                        x=pwe_teams, y=pwe_expected,
                        marker_color=[c + "66" for c in pwe_colors],
                        text=[f"{p:.0f}%" for p in pwe_expected],
                        textposition="outside",
                    ))
                    pwe_fig.add_trace(go.Bar(
                        name="Actual Win %",
                        x=pwe_teams, y=pwe_actual,
                        marker_color=pwe_colors,
                        text=[f"{p:.0f}%" for p in pwe_actual],
                        textposition="outside",
                    ))
                    apply_light_theme(pwe_fig)
                    pwe_fig.update_layout(
                        title="Expected vs Actual Win % (Pythagorean)",
                        yaxis_title="Win %", barmode="group", height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(pwe_fig, use_container_width=True)

                    # Over/underperformer callouts
                    sorted_by_pwe = sorted(rankings, key=lambda r: r.get("pwe_diff", 0), reverse=True)
                    if len(sorted_by_pwe) >= 2:
                        col1, col2 = st.columns(2)
                        top_perf = sorted_by_pwe[0]
                        bot_perf = sorted_by_pwe[-1]
                        with col1:
                            diff = top_perf.get("pwe_diff", 0) * 100
                            st.markdown(f"""
                            <div style="background:#ECFDF5; border-radius:12px; padding:16px; border-left:3px solid #10B981;">
                                <div style="color:#10B981; font-weight:700; font-size:13px;">BIGGEST OVERPERFORMER</div>
                                <div style="font-weight:700; color:{get_team_color(top_perf['team'])}; font-size:16px; margin:4px 0;">{get_short_code(top_perf['team'])}</div>
                                <div style="color:#6B7280;">Winning {diff:+.1f}% more than runs suggest</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            diff = bot_perf.get("pwe_diff", 0) * 100
                            st.markdown(f"""
                            <div style="background:#FEF2F2; border-radius:12px; padding:16px; border-left:3px solid #EF4444;">
                                <div style="color:#EF4444; font-weight:700; font-size:13px;">BIGGEST UNDERPERFORMER</div>
                                <div style="font-weight:700; color:{get_team_color(bot_perf['team'])}; font-size:16px; margin:4px 0;">{get_short_code(bot_perf['team'])}</div>
                                <div style="color:#6B7280;">Winning {diff:+.1f}% less than runs suggest</div>
                            </div>
                            """, unsafe_allow_html=True)

                # ── Elo Ratings Chart (now retention-adjusted) ──
                st.markdown("### Elo Ratings (Retention-Adjusted)")
                elo_fig = go.Figure()
                elos_sorted = [r.get("elo_rating", 1500) for r in rankings]
                elo_fig.add_trace(go.Bar(
                    x=[get_short_code(t) for t in teams_sorted],
                    y=elos_sorted,
                    marker_color=colors_sorted,
                    text=[f"{e:.0f}" for e in elos_sorted],
                    textposition="outside",
                ))
                apply_light_theme(elo_fig)
                elo_fig.update_layout(
                    title="Current Elo Ratings (adjusted for offseason squad retention)",
                    yaxis_title="Elo Rating",
                    yaxis_range=[min(elos_sorted) - 100, max(elos_sorted) + 100],
                    height=400,
                )
                st.plotly_chart(elo_fig, use_container_width=True)

                # ── Team Form States ──
                st.markdown("### Team Form (HMM Analysis)")
                st.markdown('<p style="color: #6B7280;">Hidden Markov Model detects whether a team is on a hot streak, normal form, or cold patch</p>', unsafe_allow_html=True)
                cols = st.columns(5)
                for i, r in enumerate(rankings):
                    form = r.get("form_state", "Normal")
                    form_color = {"Hot": "#00B894", "Normal": "#FDCB6E", "Cold": "#E17055"}.get(form, "#6B7280")
                    form_icon = {"Hot": "🔥", "Normal": "⚡", "Cold": "❄️"}.get(form, "⚡")
                    with cols[i % 5]:
                        st.markdown(f"""
                        <div style="text-align:center; padding:12px; background:white;
                                    border-radius:12px; border:2px solid {form_color}; margin:4px 0;">
                            <div style="font-size:24px;">{form_icon}</div>
                            <div style="font-weight:700; color:{get_team_color(r['team'])}; font-size:13px; margin:4px 0;">
                                {get_short_code(r['team'])}
                            </div>
                            <div style="font-weight:600; color:{form_color}; font-size:12px;">{form}</div>
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="info-box">Error: {e}. Train the model first: <code>python run_pipeline.py</code></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">Click <strong>Run Monte Carlo Simulation</strong> to simulate 5,000 full IPL seasons using Elo ratings.</div>', unsafe_allow_html=True)


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
    apply_light_theme(fig, 320)
    fig.update_layout(title=f"Rolling 15-Match Win Rate", yaxis=dict(range=[0, 1], title="Win Rate"))
    st.plotly_chart(fig, use_container_width=True)

    ss = tm.groupby("season").agg(played=("won", "count"), wins=("won", "sum")).reset_index()
    ss["wr"] = ss["wins"] / ss["played"]
    fig2 = go.Figure(go.Bar(
        x=ss["season"], y=ss["wr"],
        marker_color=[tc if wr >= 0.5 else "#EF4444" for wr in ss["wr"]],
        text=[f"{wr:.0%}" for wr in ss["wr"]],
        textposition="outside", textfont=dict(color="#1C1E21", size=11),
    ))
    fig2.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.1)")
    apply_light_theme(fig2, 280)
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
                textposition="outside", textfont=dict(color="#1C1E21", size=11),
            ))
            apply_light_theme(fig, max(400, len(f) * 28))
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
                textposition="outside", textfont=dict(color="#1C1E21", size=11),
            ))
            apply_light_theme(fig2, max(400, len(fb) * 28))
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
            textposition="outside", textfont=dict(color="#1C1E21", size=15, family="Inter"),
        ))
        fig.add_hline(y=65, line_dash="dash", line_color="#EF4444",
                      annotation_text="Target: 65%", annotation_font_color="#EF4444")
        apply_light_theme(fig, 350)
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
                textposition="outside", textfont=dict(color="#1C1E21"),
            ))
        apply_light_theme(fig2, 350)
        fig2.update_layout(barmode="group", yaxis_title="Accuracy %")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Feature Importance (Phase B) ──
    try:
        import joblib as _jl
        fi_path = MODELS_DIR / "feature_importance.pkl"
        if fi_path.exists():
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("### Feature Importance (XGBoost)")
            st.markdown('<p style="color: #6B7280;">Top 15 features driving predictions — includes Phase B features (Elo, HMM, Kalman)</p>', unsafe_allow_html=True)

            fi_df = _jl.load(fi_path).head(15).sort_values("importance", ascending=True)
            fi_fig = go.Figure(go.Bar(
                x=fi_df["importance"], y=fi_df["feature"], orientation="h",
                marker_color=CHART_COLORS[:len(fi_df)],
                text=[f"{v:.4f}" for v in fi_df["importance"]],
                textposition="outside", textfont=dict(color="#2D3748", size=11),
            ))
            apply_light_theme(fi_fig, max(400, len(fi_df) * 28))
            fi_fig.update_layout(
                title="Feature Importance (higher = more influential)",
                xaxis_title="Importance Score",
                margin=dict(l=200),
            )
            st.plotly_chart(fi_fig, use_container_width=True)
    except Exception:
        pass

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
        from feedback import FeedbackManager
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
                            background:#FFFFFF; border: 1px solid #E4E6EB; border-radius:10px; padding:10px 18px; margin:5px 0;">
                    <span style="font-size:16px; min-width:30px;">{medal}</span>
                    <span style="color:#1877F2; font-weight:600; min-width:80px;">{entry['session_id']}</span>
                    <span style="color:#1C1E21; font-weight:700; flex:1;">{entry['points']} pts</span>
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
