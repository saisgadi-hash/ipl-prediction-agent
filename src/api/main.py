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

from predict import predict_match, predict_tournament_winner
from team_name_mapper import ACTIVE_TEAMS

app = FastAPI(
    title="IPL Prediction Engine API",
    description="Backend API for the IPL Prediction Agent mobile app.",
    version="2.0.0",
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
    """Predict tournament winner probabilities."""
    try:
        rankings = predict_tournament_winner()
        return {"rankings": rankings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
