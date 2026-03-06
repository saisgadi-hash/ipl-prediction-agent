"""
Module B4: Bayesian Inference & Monte Carlo
Simulates an IPL tournament 10,000 times using probabilities derived from the ML model 
to estimate tournament winner probabilities.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def get_base_win_probability(team1: str, team2: str, elo_ratings: Dict[str, float]) -> float:
    """
    Very simple probability calculation based solely on Elo (for simulation).
    In a real system, you would call `model.predict_proba()` here, but 
    doing that 10,000 times per season match is computationally expensive.
    Using Elo as a proxy for the Monte Carlo is a standard fast approximation.
    """
    r1 = elo_ratings.get(team1, 1500)
    r2 = elo_ratings.get(team2, 1500)
    
    # Probability that team1 beats team2
    p1 = 1 / (1 + 10 ** ((r2 - r1) / 400.0))
    return p1

def simulate_one_season(teams: List[str], elo_ratings: Dict[str, float]) -> str:
    """
    Simulate a full double-round-robin tournament and a basic 4-team playoff.
    Returns the name of the winning team.
    """
    points = {team: 0 for team in teams}
    
    # Double round-robin
    for i, t1 in enumerate(teams):
        for j, t2 in enumerate(teams):
            if i != j:
                p1 = get_base_win_probability(t1, t2, elo_ratings)
                # Random draw
                if np.random.random() < p1:
                    points[t1] += 2
                else:
                    points[t2] += 2
                    
    # Sort teams by points (ignoring NRR for simplicity in this MVP)
    ranked_teams = sorted(points.keys(), key=lambda k: points[k], reverse=True)
    playoffs = ranked_teams[:4]
    
    # Simplistic Playoff:
    # Semi 1: 1st vs 4th
    p_sf1_t1 = get_base_win_probability(playoffs[0], playoffs[3], elo_ratings)
    sf1_winner = playoffs[0] if np.random.random() < p_sf1_t1 else playoffs[3]
    
    # Semi 2: 2nd vs 3rd
    p_sf2_t1 = get_base_win_probability(playoffs[1], playoffs[2], elo_ratings)
    sf2_winner = playoffs[1] if np.random.random() < p_sf2_t1 else playoffs[2]
    
    # Final
    p_final_t1 = get_base_win_probability(sf1_winner, sf2_winner, elo_ratings)
    champion = sf1_winner if np.random.random() < p_final_t1 else sf2_winner
    
    return champion

def simulate_tournament(teams: List[str], elo_ratings: Dict[str, float], num_simulations: int = 10000) -> Dict[str, float]:
    """
    Simulates the tournament `num_simulations` times.
    Returns a dictionary of {team: probability_of_winning_tournament}
    """
    wins = {team: 0 for team in teams}
    
    # To make it reasonably fast in Python
    for _ in range(num_simulations):
        winner = simulate_one_season(teams, elo_ratings)
        wins[winner] += 1
        
    probabilities = {t: w / num_simulations for t, w in wins.items()}
    
    # Sort descending
    return dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

if __name__ == "__main__":
    print("Testing B4 Module Monte Carlo Simulation ...")
    mock_teams = ["CSK", "MI", "RCB", "KKR", "RR", "DC", "PBKS", "SRH", "LSG", "GT"]
    mock_elo = {
        "CSK": 1600,
        "MI": 1580,
        "GT": 1550,
        "RCB": 1500,
        "RR": 1490,
        "KKR": 1480,
        "LSG": 1450,
        "PBKS": 1400,
        "DC": 1390,
        "SRH": 1380
    }
    
    print(f"Simulating 5000 seasons with {len(mock_teams)} teams...")
    results = simulate_tournament(mock_teams, mock_elo, 5000)
    
    print("\nProjected Tournament Win Probabilities:")
    for team, prob in results.items():
        print(f"{team}: {prob * 100:.1f}%")
