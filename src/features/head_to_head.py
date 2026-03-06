import pandas as pd

def calculate_team_h2h(matches_df: pd.DataFrame, team1: str, team2: str) -> dict:
    """Calculate historical head-to-head features for both teams."""
    df = matches_df[
        ((matches_df["team1"] == team1) & (matches_df["team2"] == team2)) |
        ((matches_df["team1"] == team2) & (matches_df["team2"] == team1))
    ]
    
    total = len(df)
    if total == 0:
        return {
            "h2h_matches": 0,
            "h2h_win_pct_team1": 0.5,
            "h2h_win_pct_team2": 0.5
        }
        
    t1_wins = len(df[df["winner"] == team1])
    t2_wins = len(df[df["winner"] == team2])
    
    return {
        "h2h_matches": total,
        "h2h_win_pct_team1": round(t1_wins / total, 4),
        "h2h_win_pct_team2": round(t2_wins / total, 4)
    }
