"""
IPL Prediction Engine — FastAPI Backend
Serves predictions to the Flutter mobile app and any other client.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "data_collection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "models"))

import __main__
from train_model import IPLEnsemblePredictor
__main__.IPLEnsemblePredictor = IPLEnsemblePredictor

from predict import predict_match, predict_tournament_winner, predict_live
from team_name_mapper import ACTIVE_TEAMS
from live_score_collector import LiveScoreCollector
from odds_collector import OddsCollector

# Initialize collectors (lazy — only make API calls when endpoints are hit)
_score_collector = LiveScoreCollector()
_odds_collector = OddsCollector()

app = FastAPI(
    title="IPL Prediction Engine API",
    description="Backend API for the IPL Prediction Agent mobile app.",
    version="3.0.0",
)

# Allow all origins for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response Models ──

class MatchPredictionRequest(BaseModel):
    team1: str
    team2: str
    venue: Optional[str] = None
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None

# ── Endpoints ──

@app.get("/")
def root():
    return {"status": "online", "service": "IPL Prediction Engine", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/teams")
def get_teams():
    """Return list of all active IPL teams."""
    return {"teams": sorted(list(ACTIVE_TEAMS))}

@app.post("/predict")
def predict(request: MatchPredictionRequest):
    """Predict match outcome between two teams."""
    try:
        result = predict_match(
            team1=request.team1,
            team2=request.team2,
            venue=request.venue,
            toss_winner=request.toss_winner,
            toss_decision=request.toss_decision,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tournament")
def tournament():
    """Predict tournament winner probabilities using Monte Carlo simulation."""
    try:
        rankings = predict_tournament_winner()
        return {"rankings": rankings, "method": "monte_carlo", "simulations": 5000}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/team-stats")
def team_stats():
    """Get advanced stats for all teams: Elo ratings, form states, Kalman strength."""
    try:
        from predict import _get_current_elo_ratings, _get_current_hmm_states
        teams = list(ACTIVE_TEAMS)
        elo_ratings = _get_current_elo_ratings(teams)
        hmm_states = _get_current_hmm_states(teams)
        state_labels = {0: "Cold", 1: "Normal", 2: "Hot"}

        stats = []
        for team in sorted(teams):
            stats.append({
                "team": team,
                "elo_rating": round(elo_ratings.get(team, 1500), 1),
                "form_state": state_labels.get(hmm_states.get(team, 1), "Normal"),
            })
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase D: Live Data Endpoints ──

@app.get("/live/matches")
def live_matches():
    """List current/upcoming IPL matches with live scores."""
    try:
        matches = _score_collector.get_live_matches()
        return {
            "matches": matches,
            "count": len(matches),
            "requests_remaining": _score_collector.get_requests_remaining(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/live/match/{match_id}")
def live_match(match_id: str):
    """Get live score + prediction for a specific match."""
    try:
        state = _score_collector.get_match_state(match_id)
        if not state:
            raise HTTPException(status_code=404, detail="Match not found or no live data")

        # Generate live prediction if match is in progress
        prediction = None
        if state.get("match_started") and not state.get("match_ended"):
            try:
                prediction = predict_live(state)
            except Exception:
                prediction = None

        # Get odds if available
        odds = None
        try:
            all_odds = _odds_collector.get_upcoming_odds()
            team1 = state.get("team1", "")
            team2 = state.get("team2", "")
            for o in all_odds:
                if (team1 in o.get("team1", "") or team1 in o.get("team2", "")) and \
                   (team2 in o.get("team1", "") or team2 in o.get("team2", "")):
                    odds = o
                    break
        except Exception:
            pass

        return {
            "state": state,
            "prediction": prediction,
            "odds": odds,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/live/odds")
def live_odds():
    """Get betting odds for upcoming IPL matches with model comparison."""
    try:
        odds_list = _odds_collector.get_upcoming_odds()

        # Add model predictions for comparison
        enriched = []
        for o in odds_list:
            team1 = o.get("team1", "")
            team2 = o.get("team2", "")

            # Try to get model prediction
            model_prob = None
            edge = None
            try:
                result = predict_match(team1, team2, skip_llm=True)
                if "error" not in result:
                    model_prob = result.get("team1_win_probability", 0.5)
                    implied = o.get("implied_probability_team1", 0.5)
                    edge = round((model_prob - implied) * 100, 1)
            except Exception:
                pass

            enriched.append({
                **o,
                "model_probability_team1": model_prob,
                "edge_percent": edge,
            })

        return {
            "odds": enriched,
            "count": len(enriched),
            "requests_remaining": _odds_collector.get_requests_remaining(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
