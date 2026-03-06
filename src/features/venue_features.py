import pandas as pd
import numpy as np

def calculate_venue_stats(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-computes overarching venue statistics (e.g. average first innings score)."""
    if "venue" not in deliveries_df.columns:
        merged = deliveries_df.merge(matches_df[["match_id", "venue", "city"]], on="match_id", how="left")
    else:
        merged = deliveries_df
        
    first_innings = merged[merged["innings"] == 1]
    
    venue_grouped = first_innings.groupby(["venue", "match_id"])["runs_total"].sum().reset_index()
    venue_avgs = venue_grouped.groupby("venue")["runs_total"].mean().reset_index()
    venue_avgs.rename(columns={"runs_total": "venue_avg_first_innings_runs"}, inplace=True)
    
    venue_city = matches_df[["venue", "city"]].drop_duplicates(subset=["venue"]).dropna(subset=["venue"])
    result = pd.merge(venue_avgs, venue_city, on="venue", how="left")
    
    return result

def calculate_team_venue_record(matches_df: pd.DataFrame, team: str, venue: str) -> dict:
    """Calculate the specific team's historical success rate at a specific venue."""
    df = matches_df[
        (matches_df["venue"] == venue) &
        ((matches_df["team1"] == team) | (matches_df["team2"] == team))
    ]
    
    total = len(df)
    if total == 0:
        return {"team_venue_win_pct": 0.5}
        
    wins = len(df[df["winner"] == team])
    return {"team_venue_win_pct": round(wins / total, 4)}
