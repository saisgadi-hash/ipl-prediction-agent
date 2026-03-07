from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os

# Ensure the src directory is in the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.predict import predict_match, predict_tournament_winner

app = FastAPI(
    title="IPL Prediction Engine API",
    description="Backend API for predicting IPL match and tournament outcomes for mobile apps.",
    version="1.0.0"
)

class MatchPredictionRequest(BaseModel):
    team1: str
    team2: str
    venue: Optional[str] = None
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the IPL Prediction API. Access /docs for documentation."}

@app.post("/predict_match")
def get_match_prediction(request: MatchPredictionRequest):
    """
    Predict the outcome of a match between two teams.
    """
    try:
        result = predict_match(
            team1=request.team1,
            team2=request.team2,
            venue=request.venue,
            toss_winner=request.toss_winner,
            toss_decision=request.toss_decision
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict_tournament")
def get_tournament_prediction():
    """
    Predict the tournament winner probabilities based on 10,000 Monte Carlo simulations.
    """
    try:
        rankings = predict_tournament_winner()
        return {"rankings": rankings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
