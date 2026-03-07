"""
Module B3: Elo Ratings and Kalman Filter
Calculates true Elo ratings and Kalman Filter hidden team strength based on historical match results.

Phase E: Offseason retention decay — Elo ratings are decayed toward
1500 at season boundaries proportional to squad turnover.
"""
import pandas as pd
import numpy as np

try:
    from retention_decay import apply_retention_decay, get_retention_pct
except ImportError:
    # Fallback if retention_decay not available
    def apply_retention_decay(elo, pct, **kw):
        return elo
    def get_retention_pct(team, prev, curr):
        return 0.45


def calculate_elo(matches: pd.DataFrame, team: str, current_date, k: int = 32, start_elo: float = 1500.0) -> float:
    """
    Computes Elo rating for a team by chronologically playing through all
    historical matches before the given current_date.

    Phase E: Applies offseason retention decay at season boundaries.

    Returns the final Elo rating of the team.
    """
    past_matches = matches[matches['date'] < current_date].sort_values('date')

    # Track ratings for all teams that appear, initialized to start_elo
    elo_dict = {}
    prev_season = None

    for _, match in past_matches.iterrows():
        t1 = match.get('team1')
        t2 = match.get('team2')
        winner = match.get('winner')
        match_season = str(match.get('season', ''))

        if not t1 or not t2 or not winner or winner in ['no result', 'tie']:
            continue

        # Phase E: Detect season boundary and apply retention decay
        if prev_season and match_season and match_season != prev_season:
            for t in list(elo_dict.keys()):
                retention = get_retention_pct(t, prev_season, match_season)
                elo_dict[t] = apply_retention_decay(elo_dict[t], retention)

        prev_season = match_season

        # Initialize if new
        if t1 not in elo_dict:
            elo_dict[t1] = start_elo
        if t2 not in elo_dict:
            elo_dict[t2] = start_elo

        r1 = elo_dict[t1]
        r2 = elo_dict[t2]

        # Expected scores
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400.0))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400.0))

        # Actual scores
        s1 = 1.0 if winner == t1 else 0.0
        s2 = 1.0 if winner == t2 else 0.0

        # Update ratings
        elo_dict[t1] = r1 + k * (s1 - e1)
        elo_dict[t2] = r2 + k * (s2 - e2)

    return elo_dict.get(team, start_elo)


def estimate_kalman_strength(matches: pd.DataFrame, team: str, current_date, 
                             initial_strength: float = 0.5, 
                             process_noise: float = 0.01, 
                             measurement_noise: float = 0.1) -> float:
    """
    A simplified 1D Kalman Filter to estimate 'true team strength' vs expected performance.
    
    - initial_strength: State estimate start (e.g. 0.5 for neutral)
    - process_noise (Q): How much true strength naturally varies match-to-match
    - measurement_noise (R): How much luck/noise is in a single match result
    
    Strength is represented on a 0.0 to 1.0 scale.
    """
    past_matches = matches[
        ((matches['team1'] == team) | (matches['team2'] == team)) & 
        (matches['date'] < current_date)
    ].sort_values('date')
    
    # State (estimated strength) and uncertainty (variance)
    x = initial_strength
    p = 1.0  # Initial high uncertainty
    
    for _, match in past_matches.iterrows():
        winner = match.get('winner')
        if not winner or winner in ['no result', 'tie']:
            continue
            
        # Measurement: 1.0 if win, 0.0 if loss
        z = 1.0 if winner == team else 0.0
        
        # 1. Predict
        x_pred = x  # Assume strength stays same
        p_pred = p + process_noise
        
        # 2. Update (Measurement)
        y = z - x_pred  # Residual
        S = p_pred + measurement_noise  # Innovation covariance
        K = p_pred / S  # Kalman Gain
        
        x = x_pred + K * y
        p = (1 - K) * p_pred
        
    # Bound between 0.0 and 1.0
    return max(0.0, min(1.0, x))

if __name__ == "__main__":
    print("Testing B3 Module Elo & Kalman ...")
    # Small test mock
    mock_data = {
        'date': pd.to_datetime(['2024-04-01', '2024-04-05', '2024-04-10']),
        'team1': ['CSK', 'MI', 'CSK'],
        'team2': ['RCB', 'CSK', 'RR'],
        'winner': ['CSK', 'CSK', 'RR']
    }
    df = pd.DataFrame(mock_data)
    
    csk_elo = calculate_elo(df, 'CSK', pd.to_datetime('2024-04-15'))
    csk_kalman = estimate_kalman_strength(df, 'CSK', pd.to_datetime('2024-04-15'))
    
    print(f"CSK Elo: {csk_elo:.1f} | CSK Strength: {csk_kalman:.3f}")
