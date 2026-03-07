"""
Phase E: Poisson Binomial Distribution & Confidence Intervals

In a tournament, each match has a DIFFERENT win probability for each team
(e.g., 72% vs CSK, 55% vs MI, 48% vs RCB). The Poisson Binomial Distribution
gives the exact probability of winning exactly k out of n matches when each
match has a different success probability.

This replaces the Normal approximation used in basic Monte Carlo simulations
and gives proper confidence intervals for tournament predictions.

Algorithm: Recursive Dynamic Programming
    dp[i][k] = P(team wins exactly k out of first i matches)
    dp[i+1][k] = dp[i][k] * (1 - p[i+1]) + dp[i][k-1] * p[i+1]
    Time: O(n^2), Space: O(n)

HOW TO USE:
    from confidence_intervals import compute_confidence_interval
    ci = compute_confidence_interval([0.72, 0.55, 0.48, 0.63, ...])
    # ci = {"mean": 8.2, "ci_lower": 6, "ci_upper": 10, "std": 1.4, ...}
"""

import numpy as np


def poisson_binomial_pmf(probabilities: list) -> np.ndarray:
    """
    Compute the exact PMF of the Poisson Binomial Distribution using DP.

    Given n independent Bernoulli trials with different success probabilities
    p_1, p_2, ..., p_n, computes P(X = k) for k = 0, 1, ..., n.

    Args:
        probabilities: List of success probabilities for each trial (match)

    Returns:
        numpy array of length n+1 where pmf[k] = P(winning exactly k matches)
    """
    n = len(probabilities)
    if n == 0:
        return np.array([1.0])

    # dp[k] = probability of exactly k successes so far
    dp = np.zeros(n + 1)
    dp[0] = 1.0

    for i, p in enumerate(probabilities):
        # Process in reverse to avoid overwriting needed values
        new_dp = np.zeros(n + 1)
        for k in range(i + 2):  # k can be at most i+1
            if k == 0:
                new_dp[0] = dp[0] * (1 - p)
            else:
                new_dp[k] = dp[k] * (1 - p) + dp[k - 1] * p
        dp = new_dp

    return dp


def compute_confidence_interval(
    probabilities: list,
    confidence: float = 0.95,
) -> dict:
    """
    Compute the confidence interval for total wins using Poisson Binomial.

    Args:
        probabilities: List of per-match win probabilities
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        dict with:
            mean: Expected number of wins
            std: Standard deviation
            ci_lower: Lower bound of CI (number of wins)
            ci_upper: Upper bound of CI (number of wins)
            ci_lower_pct: Lower bound as fraction of total matches
            ci_upper_pct: Upper bound as fraction of total matches
            confidence: The confidence level used
    """
    n = len(probabilities)
    if n == 0:
        return {
            "mean": 0, "std": 0,
            "ci_lower": 0, "ci_upper": 0,
            "ci_lower_pct": 0.0, "ci_upper_pct": 0.0,
            "confidence": confidence,
        }

    # Mean and variance (analytical — always exact)
    mean = sum(probabilities)
    variance = sum(p * (1 - p) for p in probabilities)
    std = variance ** 0.5

    # For small n (≤ 50 matches, typical IPL season), use exact DP
    # For large n, fall back to Normal approximation
    if n <= 60:
        pmf = poisson_binomial_pmf(probabilities)
        cdf = np.cumsum(pmf)

        # Find CI bounds from CDF
        alpha = (1 - confidence) / 2
        ci_lower = 0
        ci_upper = n

        for k in range(n + 1):
            if cdf[k] >= alpha:
                ci_lower = k
                break

        for k in range(n + 1):
            if cdf[k] >= 1 - alpha:
                ci_upper = k
                break
    else:
        # Normal approximation for large n
        from scipy.stats import norm
        z = norm.ppf(1 - (1 - confidence) / 2)
        ci_lower = max(0, int(np.floor(mean - z * std)))
        ci_upper = min(n, int(np.ceil(mean + z * std)))

    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_lower_pct": round(ci_lower / max(n, 1), 4),
        "ci_upper_pct": round(ci_upper / max(n, 1), 4),
        "confidence": confidence,
    }


def tournament_confidence_bands(
    teams: list,
    elo_ratings: dict,
    num_matches_per_team: int = 14,
) -> dict:
    """
    Compute confidence bands for tournament win probabilities.

    For each team, generates the per-match win probabilities against
    all opponents (based on Elo), then uses Poisson Binomial to compute
    the 95% CI for total league-stage wins.

    Also computes CI for tournament win probability from Monte Carlo.

    Args:
        teams: List of team names
        elo_ratings: Dict of {team: elo_rating}
        num_matches_per_team: Matches per team in league stage (default 14)

    Returns:
        dict of {team: {mean_wins, ci_lower, ci_upper, ci_lower_pct, ci_upper_pct, match_probs}}
    """
    results = {}

    for team in teams:
        team_elo = elo_ratings.get(team, 1500.0)

        # Generate per-match win probabilities against each opponent
        opponents = [t for t in teams if t != team]
        match_probs = []

        for opp in opponents:
            opp_elo = elo_ratings.get(opp, 1500.0)
            # Elo expected score formula
            expected = 1.0 / (1.0 + 10 ** ((opp_elo - team_elo) / 400.0))
            match_probs.append(expected)

        # In IPL, each team plays ~14 matches (home+away vs some opponents)
        # Scale to num_matches by sampling with replacement
        if len(match_probs) > 0:
            # Double round-robin: play each opponent twice
            double_probs = match_probs * 2  # 18 matches (9 opponents × 2)
            # Take first num_matches_per_team
            season_probs = double_probs[:num_matches_per_team]
        else:
            season_probs = [0.5] * num_matches_per_team

        # Compute CI
        ci = compute_confidence_interval(season_probs, confidence=0.95)
        ci["match_probs"] = season_probs

        results[team] = ci

    return results


def monte_carlo_confidence_interval(
    win_counts: list,
    num_simulations: int,
    confidence: float = 0.95,
) -> tuple:
    """
    Compute confidence interval for a tournament win probability
    from Monte Carlo simulation results using the Wilson score interval.

    This is more accurate than the simple p ± z*sqrt(p(1-p)/n) for
    proportions near 0 or 1.

    Args:
        win_counts: Number of tournament wins for this team
        num_simulations: Total simulations run
        confidence: Confidence level

    Returns:
        (probability, ci_lower, ci_upper)
    """
    from scipy.stats import norm

    if isinstance(win_counts, list):
        wins = sum(win_counts)
    else:
        wins = win_counts

    p = wins / max(num_simulations, 1)
    z = norm.ppf(1 - (1 - confidence) / 2)
    n = num_simulations

    # Wilson score interval (handles edge cases better)
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator

    ci_lower = max(0.0, centre - margin)
    ci_upper = min(1.0, centre + margin)

    return (round(p, 4), round(ci_lower, 4), round(ci_upper, 4))


if __name__ == "__main__":
    print("Confidence Intervals module loaded.")
    print("\nExample: Team with match probabilities [0.7, 0.6, 0.55, 0.5, 0.45]")
    probs = [0.7, 0.6, 0.55, 0.5, 0.45]
    ci = compute_confidence_interval(probs)
    print(f"  Expected wins: {ci['mean']:.1f} ± {ci['std']:.1f}")
    print(f"  95% CI: [{ci['ci_lower']}, {ci['ci_upper']}] wins out of {len(probs)}")

    print("\nMonte Carlo CI: 1600 wins out of 5000 sims")
    p, lo, hi = monte_carlo_confidence_interval(1600, 5000)
    print(f"  Win probability: {p:.1%} [{lo:.1%} - {hi:.1%}]")
