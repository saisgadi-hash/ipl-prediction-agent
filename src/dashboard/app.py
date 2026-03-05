"""
IPL Prediction Agent — Main Streamlit Dashboard

This is the entry point for the web dashboard. It provides:
- Match predictions with win probabilities
- Tournament winner rankings
- Team strength comparisons
- Historical accuracy tracking
- Interactive what-if scenarios

HOW TO RUN:
    streamlit run src/dashboard/app.py

BEGINNER NOTES:
    - Streamlit turns Python scripts into web apps automatically
    - st.sidebar creates a left sidebar for controls
    - st.columns creates side-by-side layout
    - st.metric shows a big number with optional delta
    - Plotly creates interactive charts (hover, zoom, etc.)
"""

import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Page Configuration ──
st.set_page_config(
    page_title="IPL Prediction Agent",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; border-radius: 10px; padding: 15px; }
    .big-number { font-size: 48px; font-weight: bold; color: #1F4E79; }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px; padding: 20px; color: white; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load all required data files."""
    data = {}

    # Try to load matches
    matches_path = os.path.join("data", "processed", "matches.csv")
    if os.path.exists(matches_path):
        data["matches"] = pd.read_csv(matches_path, parse_dates=["date"])
    else:
        data["matches"] = None

    # Try to load training metadata
    metadata_path = os.path.join("models", "training_metadata.pkl")
    if os.path.exists(metadata_path):
        import joblib
        data["metadata"] = joblib.load(metadata_path)
    else:
        data["metadata"] = None

    # Try to load backtest results
    backtest_path = os.path.join("data", "processed", "backtest_results.csv")
    if os.path.exists(backtest_path):
        data["backtest"] = pd.read_csv(backtest_path)
    else:
        data["backtest"] = None

    return data


def main():
    """Main dashboard layout."""

    # ── Sidebar ──
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/200px-Indian_Premier_League_Official_Logo.svg.png", width=150)
    st.sidebar.title("IPL Prediction Agent")
    st.sidebar.markdown("*AI-powered match predictions*")
    st.sidebar.divider()

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Dashboard", "🔮 Match Predictor", "🏆 Tournament Rankings",
         "📊 Team Analysis", "📈 Model Performance", "🧪 What-If Scenarios"],
    )

    # Load data
    data = load_data()

    # ── Route to pages ──
    if page == "🏠 Dashboard":
        show_dashboard(data)
    elif page == "🔮 Match Predictor":
        show_match_predictor(data)
    elif page == "🏆 Tournament Rankings":
        show_tournament_rankings(data)
    elif page == "📊 Team Analysis":
        show_team_analysis(data)
    elif page == "📈 Model Performance":
        show_model_performance(data)
    elif page == "🧪 What-If Scenarios":
        show_what_if(data)


def show_dashboard(data):
    """Main dashboard overview page."""
    st.title("🏏 IPL Prediction Agent Dashboard")
    st.markdown("Real-time AI predictions updated after every match")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    matches = data.get("matches")
    metadata = data.get("metadata")

    with col1:
        total_matches = len(matches) if matches is not None else 0
        st.metric("Matches Analysed", f"{total_matches:,}")

    with col2:
        accuracy = metadata.get("ensemble_accuracy", 0) * 100 if metadata else 0
        st.metric("Model Accuracy", f"{accuracy:.1f}%")

    with col3:
        features = metadata.get("feature_count", 0) if metadata else 0
        st.metric("Features Used", f"{features}")

    with col4:
        if matches is not None:
            seasons = matches["season"].nunique()
            st.metric("Seasons Covered", f"{seasons}")

    st.divider()

    if matches is not None and len(matches) > 0:
        # Recent matches
        st.subheader("Recent Match Results")
        recent = matches.sort_values("date", ascending=False).head(10)
        for _, match in recent.iterrows():
            col1, col2, col3 = st.columns([3, 1, 3])
            with col1:
                winner_emoji = "🏆" if match["winner"] == match["team1"] else ""
                st.write(f"**{match['team1']}** {winner_emoji}")
            with col2:
                st.write(f"vs")
            with col3:
                winner_emoji = "🏆" if match["winner"] == match["team2"] else ""
                st.write(f"**{match['team2']}** {winner_emoji}")

        # Win distribution chart
        st.subheader("All-Time Win Distribution")
        if "winner" in matches.columns:
            win_counts = matches["winner"].value_counts().head(10)
            fig = px.bar(
                x=win_counts.values, y=win_counts.index,
                orientation="h", color=win_counts.values,
                color_continuous_scale="viridis",
                labels={"x": "Wins", "y": "Team"},
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No match data loaded yet. Run the data collection pipeline first:\n\n"
            "```bash\n"
            "python -m src.data_collection.download_cricsheet\n"
            "python -m src.data_collection.parse_matches\n"
            "```"
        )


def show_match_predictor(data):
    """Interactive match prediction page."""
    st.title("🔮 Match Predictor")
    st.markdown("Select two teams to predict the match outcome")

    teams = [
        "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
        "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
        "Rajasthan Royals", "Sunrisers Hyderabad", "Lucknow Super Giants",
        "Gujarat Titans",
    ]

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, index=0)
    with col2:
        team2 = st.selectbox("Team 2", [t for t in teams if t != team1], index=0)

    col3, col4 = st.columns(2)
    with col3:
        toss_winner = st.selectbox("Toss Winner", ["Unknown", team1, team2])
    with col4:
        toss_decision = st.selectbox("Toss Decision", ["Unknown", "Bat First", "Field First"])

    if st.button("🔮 Predict Winner", type="primary", use_container_width=True):
        try:
            from src.models.predict import predict_match
            result = predict_match(
                team1, team2,
                toss_winner=toss_winner if toss_winner != "Unknown" else None,
                toss_decision=toss_decision.split()[0].lower() if toss_decision != "Unknown" else None,
            )

            if "error" not in result:
                # Display prediction
                st.divider()
                col1, col2, col3 = st.columns([2, 1, 2])

                prob1 = result["team1_win_probability"]
                prob2 = result["team2_win_probability"]

                with col1:
                    st.subheader(team1)
                    st.markdown(f"### {prob1:.1%}")
                    st.progress(prob1)

                with col2:
                    st.markdown("### VS")

                with col3:
                    st.subheader(team2)
                    st.markdown(f"### {prob2:.1%}")
                    st.progress(prob2)

                st.success(f"🏆 Predicted Winner: **{result['predicted_winner']}** (Confidence: {result['confidence']:.0f}%)")

                # Win probability pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=[team1, team2],
                    values=[prob1, prob2],
                    hole=0.4,
                    marker_colors=["#667eea", "#764ba2"],
                )])
                fig.update_layout(height=300, title="Win Probability")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(result["error"])
        except Exception as e:
            st.warning(f"Model not trained yet. Train the model first, then predictions will work here.\n\nError: {e}")


def show_tournament_rankings(data):
    """Tournament winner predictions."""
    st.title("🏆 Tournament Winner Predictions")

    if st.button("Generate Rankings", type="primary"):
        try:
            from src.models.predict import predict_tournament_winner
            with st.spinner("Calculating tournament probabilities..."):
                rankings = predict_tournament_winner()

            if rankings:
                for r in rankings:
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col1:
                        st.markdown(f"### #{r['rank']}")
                    with col2:
                        st.write(f"**{r['team']}**")
                        st.progress(r["win_probability"])
                    with col3:
                        st.markdown(f"### {r['win_probability']:.1%}")
        except Exception as e:
            st.warning(f"Model not trained yet. Error: {e}")
    else:
        st.info("Click 'Generate Rankings' to calculate tournament winner probabilities.")


def show_team_analysis(data):
    """Team comparison and analysis."""
    st.title("📊 Team Analysis")

    matches = data.get("matches")
    if matches is None:
        st.info("Load match data first.")
        return

    teams = sorted(matches["team1"].unique())
    selected_team = st.selectbox("Select Team", teams)

    # Team's match history
    team_matches = matches[
        (matches["team1"] == selected_team) | (matches["team2"] == selected_team)
    ].sort_values("date")

    team_matches["won"] = (team_matches["winner"] == selected_team).astype(int)

    # Win rate over time (rolling)
    team_matches["rolling_win_rate"] = team_matches["won"].rolling(10, min_periods=1).mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Matches", len(team_matches))
    with col2:
        wins = team_matches["won"].sum()
        st.metric("Wins", int(wins))
    with col3:
        st.metric("Win Rate", f"{wins / max(len(team_matches), 1):.1%}")

    # Rolling win rate chart
    fig = px.line(
        team_matches, x="date", y="rolling_win_rate",
        title=f"{selected_team} — Rolling 10-Match Win Rate",
        labels={"rolling_win_rate": "Win Rate", "date": "Date"},
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="50%")
    st.plotly_chart(fig, use_container_width=True)

    # Season-wise breakdown
    season_stats = team_matches.groupby("season").agg(
        matches=("won", "count"),
        wins=("won", "sum"),
    ).reset_index()
    season_stats["win_rate"] = season_stats["wins"] / season_stats["matches"]

    fig2 = px.bar(
        season_stats, x="season", y="win_rate",
        title=f"{selected_team} — Win Rate by Season",
        labels={"win_rate": "Win Rate", "season": "Season"},
        color="win_rate", color_continuous_scale="RdYlGn",
    )
    fig2.add_hline(y=0.5, line_dash="dash", line_color="gray")
    st.plotly_chart(fig2, use_container_width=True)


def show_model_performance(data):
    """Model accuracy and performance tracking."""
    st.title("📈 Model Performance")

    metadata = data.get("metadata")
    backtest = data.get("backtest")

    if metadata:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("XGBoost", f"{metadata.get('xgboost_accuracy', 0):.1%}")
        with col2:
            st.metric("LightGBM", f"{metadata.get('lightgbm_accuracy', 0):.1%}")
        with col3:
            st.metric("Logistic", f"{metadata.get('logistic_accuracy', 0):.1%}")
        with col4:
            st.metric("Ensemble", f"{metadata.get('ensemble_accuracy', 0):.1%}", delta="Best")

    if backtest is not None:
        st.subheader("Back-Test Results by Season")
        fig = px.bar(
            backtest, x="season", y="accuracy", color="adaptive",
            barmode="group",
            labels={"accuracy": "Accuracy", "season": "Season", "adaptive": "Adaptive Learning"},
            title="Static vs Adaptive Model Accuracy per Season",
        )
        fig.add_hline(y=0.65, line_dash="dash", annotation_text="Target: 65%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run back-testing to see performance: `python -m src.models.backtest`")


def show_what_if(data):
    """Interactive what-if scenario tool."""
    st.title("🧪 What-If Scenarios")
    st.markdown("Explore how different factors change the prediction")

    teams = [
        "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
        "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
        "Rajasthan Royals", "Sunrisers Hyderabad", "Lucknow Super Giants",
        "Gujarat Titans",
    ]

    team1 = st.selectbox("Team 1", teams, key="wif_t1")
    team2 = st.selectbox("Team 2", [t for t in teams if t != team1], key="wif_t2")

    st.subheader("Adjust Scenarios")

    col1, col2 = st.columns(2)
    with col1:
        key_player_out = st.checkbox("Key player injured (Team 1)")
        home_advantage = st.checkbox("Team 1 playing at home")
    with col2:
        dew_factor = st.slider("Dew Factor (0=none, 10=heavy)", 0, 10, 5)
        team1_form = st.slider("Team 1 Recent Form (wins in last 5)", 0, 5, 3)

    if st.button("Simulate Scenario", type="primary"):
        st.info(
            "Scenario simulation requires a trained model.\n\n"
            "Once trained, this tool adjusts the input features based on your selections "
            "and re-runs the prediction to show how each factor affects the outcome."
        )


if __name__ == "__main__":
    main()
