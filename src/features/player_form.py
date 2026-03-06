import pandas as pd
import numpy as np

def calculate_batting_form(past_deliveries: pd.DataFrame, batter: str, window: int = 10) -> dict:
    """Calculate the recent batting form for a given batter based on recent matches."""
    df = past_deliveries[past_deliveries["batter"] == batter]
    if len(df) == 0:
        return {"form_index": 0.0}
    
    matches_played = df["match_id"].unique()
    recent_matches = sorted(matches_played)[-window:]
    recent_df = df[df["match_id"].isin(recent_matches)]
    
    if len(recent_df) == 0:
        return {"form_index": 0.0}

    runs = recent_df["runs_batter"].sum()
    dismissals = len(recent_df[(recent_df["is_wicket"] == 1) & (recent_df["wicket_player"] == batter)])
    balls = len(recent_df[(recent_df["extras_wides"] == 0)])
    
    avg = runs / dismissals if dismissals > 0 else runs
    sr = runs / balls * 100 if balls > 0 else 0
    
    # Simple form heuristic: A balance of average and strike rate.
    # We assign slightly more weight to consistency (average).
    form_index = (avg * 1.5) + (sr / 2.0)
    
    return {"form_index": form_index}

def calculate_bowling_form(past_deliveries: pd.DataFrame, bowler: str, window: int = 10) -> dict:
    """Calculate the recent bowling form for a given bowler based on recent matches."""
    df = past_deliveries[past_deliveries["bowler"] == bowler]
    if len(df) == 0:
        return {"form_index": 0.0}
    
    matches_played = df["match_id"].unique()
    recent_matches = sorted(matches_played)[-window:]
    recent_df = df[df["match_id"].isin(recent_matches)]
    
    if len(recent_df) == 0:
        return {"form_index": 0.0}

    # Wickets avoiding non-bowler wickets
    mask = (recent_df["is_wicket"] == 1) & (~recent_df["wicket_kind"].isin(["run out", "retired hurt", "obstructing the field"]))
    wickets = len(recent_df[mask])
        
    runs_conceded = recent_df["runs_total"].sum() - recent_df["extras_legbyes"].sum() - recent_df["extras_byes"].sum() 
    legal_balls = len(recent_df[(recent_df["extras_wides"] == 0) & (recent_df["extras_noballs"] == 0)])
    
    econ = runs_conceded / (legal_balls / 6.0) if legal_balls > 0 else 0
    strike_rate = legal_balls / wickets if wickets > 0 else legal_balls
    
    # Form index heuristic (higher is better):
    # Base for taking wickets minus the penalty for poor economy.
    form_index = (wickets * 15) - (econ * 2.0)
    
    return {"form_index": form_index}
