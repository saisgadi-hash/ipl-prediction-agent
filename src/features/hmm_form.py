"""
Module B5: Hidden Markov Models
Uses hmmlearn to deduce a team's hidden form state (hot, normal, cold) based on historical sequence of results.
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm

def calculate_hmm_state(matches: pd.DataFrame, team: str, current_date, min_history: int = 5) -> int:
    """
    Computes the most likely hidden state of a team's form (0=Cold, 1=Normal, 2=Hot) 
    at a specific current_date by training an HMM on their historical match results.
    
    Returns 0, 1, or 2 representing the state. Default 1 (Normal) if not enough history.
    """
    past_matches = matches[
        ((matches['team1'] == team) | (matches['team2'] == team)) & 
        (matches['date'] < current_date)
    ].sort_values('date')
    
    if len(past_matches) < min_history:
        return 1  # Default to normal state
        
    # Extract sequence of wins (1) and losses (0)
    results = []
    for _, match in past_matches.iterrows():
        winner = match.get('winner')
        if winner == team:
            results.append(1)
        elif winner not in ['no result', 'tie']:
            results.append(0)
            
    if len(results) < min_history:
        return 1
        
    sequence = np.array(results).reshape(-1, 1)
    
    # Define a 3-state HMM
    # We initialize the model with explicit parameters to bias it for sports form
    # Cold (0), Normal (1), Hot (2)
    model = hmm.CategoricalHMM(n_components=3, random_state=42, n_iter=100)
    
    try:
        model.fit(sequence)
        
        # Predict the most likely sequence of hidden states
        hidden_states = model.predict(sequence)
        
        # We need to map the hidden states to meaning based on their emission probabilities.
        # State with highest probability of emitting '1' (win) is 'Hot'.
        # State with lowest probability is 'Cold'.
        
        # emissionprob_ shape: (n_components, n_features) -> (3, 2)
        win_probs = model.emissionprob_[:, 1]
        state_meanings = np.argsort(win_probs)  # Indices of lowest, middle, highest win probability
        
        # The true state mapping:
        # state_meanings[0] is the internal ID for "Cold"
        # state_meanings[1] is the internal ID for "Normal"
        # state_meanings[2] is the internal ID for "Hot"
        
        # Get the internal ID of the final state in the sequence
        current_internal_state = hidden_states[-1]
        
        # Map internal state to our 0, 1, 2 semantic scale
        if current_internal_state == state_meanings[0]:
            return 0  # Cold
        elif current_internal_state == state_meanings[2]:
            return 2  # Hot
        else:
            return 1  # Normal
            
    except Exception as e:
        print(f"HMM Error for {team}: {e}. Defaulting to Normal.")
        return 1

if __name__ == "__main__":
    print("Testing B5 Module HMM Form ...")
    # Small test mock
    mock_data = {
        'date': pd.date_range(start='1/1/2024', periods=10, freq='D'),
        'team1': ['CSK']*10,
        'team2': ['RR']*10,
        'winner': ['CSK', 'CSK', 'CSK', 'RR', 'RR', 'RR', 'CSK', 'CSK', 'CSK', 'CSK']
    }
    df = pd.DataFrame(mock_data)
    
    # Expect Hot state after a 4-game win streak
    state = calculate_hmm_state(df, 'CSK', pd.to_datetime('2024-01-15'))
    print(f"CSK HMM State (0=Cold, 1=Normal, 2=Hot): {state}")
