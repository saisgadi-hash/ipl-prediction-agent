"""
Phase E: Offseason Retention Decay

IPL teams change rosters significantly between seasons through mega auctions
and retention policies. A team's Elo rating from last season should be
decayed toward the baseline (1500) proportional to how many players they lost.

If a team retained 70% of their squad, their Elo advantage carries forward
strongly. If they only retained 30% (e.g., mega auction rebuild), their Elo
is pulled much closer to 1500.

Formula:
    adjusted_elo = 1500 + (old_elo - 1500) * decay_multiplier
    where decay_multiplier = retention_pct * (1 - base_decay) + base_decay
    and base_decay = 0.3 (minimum carry-forward even with 0% retention)

HOW TO USE:
    from retention_decay import apply_retention_decay, get_retention_pct
    retention = get_retention_pct("Chennai Super Kings", "2023", "2024")
    new_elo = apply_retention_decay(old_elo=1580, retention_pct=retention)
"""


# ── Estimated Retention Percentages ──
# Based on IPL mega auction years (2018, 2022) and regular retention windows.
# Values represent approximate % of playing-XI core retained.
# Mega auction years show much lower retention across the board.
# Format: { "from_season/to_season": { team: retention_pct } }

RETENTION_DATA = {
    # 2017 → 2018: MEGA AUCTION (only 3-5 players retained per team)
    "2017/2018": {
        "Chennai Super Kings": 0.45,  # Returned from ban, retained Dhoni, Raina, Jadeja
        "Mumbai Indians": 0.40,       # Retained Rohit, Bumrah, Pandya
        "Royal Challengers Bengaluru": 0.35,  # Retained Kohli, de Villiers
        "Kolkata Knight Riders": 0.35,
        "Delhi Capitals": 0.30,
        "Punjab Kings": 0.30,
        "Rajasthan Royals": 0.40,     # Returned from ban
        "Sunrisers Hyderabad": 0.40,  # Retained Warner, Williamson
    },
    # 2018 → 2019: Normal retention
    "2018/2019": {
        "Chennai Super Kings": 0.75,
        "Mumbai Indians": 0.75,
        "Royal Challengers Bengaluru": 0.65,
        "Kolkata Knight Riders": 0.70,
        "Delhi Capitals": 0.60,
        "Punjab Kings": 0.55,
        "Rajasthan Royals": 0.65,
        "Sunrisers Hyderabad": 0.70,
    },
    # 2019 → 2020: Normal retention
    "2019/2020": {
        "Chennai Super Kings": 0.70,
        "Mumbai Indians": 0.80,
        "Royal Challengers Bengaluru": 0.60,
        "Kolkata Knight Riders": 0.65,
        "Delhi Capitals": 0.70,
        "Punjab Kings": 0.55,
        "Rajasthan Royals": 0.60,
        "Sunrisers Hyderabad": 0.65,
    },
    # 2020 → 2021: Normal retention
    "2020/2021": {
        "Chennai Super Kings": 0.70,
        "Mumbai Indians": 0.75,
        "Royal Challengers Bengaluru": 0.65,
        "Kolkata Knight Riders": 0.60,
        "Delhi Capitals": 0.75,
        "Punjab Kings": 0.55,
        "Rajasthan Royals": 0.60,
        "Sunrisers Hyderabad": 0.60,
    },
    # 2021 → 2022: MEGA AUCTION (expanded to 10 teams, max 4 retentions)
    "2021/2022": {
        "Chennai Super Kings": 0.35,  # Retained Dhoni, Jadeja, Gaikwad, Moeen
        "Mumbai Indians": 0.35,       # Retained Rohit, Bumrah, Suryakumar, Kishan
        "Royal Challengers Bengaluru": 0.30,  # Retained Kohli, Siraj, Faf
        "Kolkata Knight Riders": 0.25,
        "Delhi Capitals": 0.35,       # Retained Pant, Shaw, Axar, Nortje
        "Punjab Kings": 0.25,
        "Rajasthan Royals": 0.30,
        "Sunrisers Hyderabad": 0.25,
        "Lucknow Super Giants": 0.00, # New team
        "Gujarat Titans": 0.00,       # New team
    },
    # 2022 → 2023: Normal retention
    "2022/2023": {
        "Chennai Super Kings": 0.70,
        "Mumbai Indians": 0.65,
        "Royal Challengers Bengaluru": 0.65,
        "Kolkata Knight Riders": 0.60,
        "Delhi Capitals": 0.70,
        "Punjab Kings": 0.55,
        "Rajasthan Royals": 0.70,
        "Sunrisers Hyderabad": 0.55,
        "Lucknow Super Giants": 0.65,
        "Gujarat Titans": 0.75,
    },
    # 2023 → 2024: Normal retention
    "2023/2024": {
        "Chennai Super Kings": 0.65,
        "Mumbai Indians": 0.70,
        "Royal Challengers Bengaluru": 0.60,
        "Kolkata Knight Riders": 0.55,
        "Delhi Capitals": 0.60,
        "Punjab Kings": 0.50,
        "Rajasthan Royals": 0.70,
        "Sunrisers Hyderabad": 0.55,
        "Lucknow Super Giants": 0.60,
        "Gujarat Titans": 0.65,
    },
    # 2024 → 2025: MEGA AUCTION (max 6 retentions, new retention rules)
    "2024/2025": {
        "Chennai Super Kings": 0.50,  # Retained Gaikwad, Jadeja, Dube, Pathirana, Dhoni
        "Mumbai Indians": 0.45,       # Retained Hardik, Suryakumar, Bumrah, Brevis, Tilak
        "Royal Challengers Bengaluru": 0.40,  # Retained Kohli, du Plessis, Rajat Patidar
        "Kolkata Knight Riders": 0.50, # Retained Rinku, Sunil Narine, Varun, Harshit, Ramandeep
        "Delhi Capitals": 0.35,       # Retained Axar, Kuldeep
        "Punjab Kings": 0.25,         # Almost full rebuild
        "Rajasthan Royals": 0.45,     # Retained Sanju, Yashasvi, Chahal
        "Sunrisers Hyderabad": 0.45,  # Retained Travis Head, Pat Cummins, Abhishek, Nitish
        "Lucknow Super Giants": 0.40, # Retained Nicholas Pooran, Ravi Bishnoi
        "Gujarat Titans": 0.40,       # Retained Rashid Khan, Shubman Gill, Sai Sudharsan
    },
}

# Default retention for seasons not in lookup
DEFAULT_RETENTION = 0.45


def get_retention_pct(team: str, prev_season: str, curr_season: str) -> float:
    """
    Look up the estimated retention percentage for a team between two seasons.

    Args:
        team: Team name (standardised)
        prev_season: e.g., "2023"
        curr_season: e.g., "2024"

    Returns:
        Retention percentage (0.0 to 1.0)
    """
    key = f"{prev_season}/{curr_season}"
    season_data = RETENTION_DATA.get(key, {})
    return season_data.get(team, DEFAULT_RETENTION)


def apply_retention_decay(
    old_elo: float,
    retention_pct: float,
    base_decay: float = 0.3,
    baseline_elo: float = 1500.0,
) -> float:
    """
    Apply offseason retention decay to an Elo rating.

    The Elo is pulled toward baseline proportional to squad turnover.
    Even with 0% retention, 30% of the Elo advantage carries forward
    (because coaching, culture, and infrastructure persist).

    Args:
        old_elo: Team's Elo at end of previous season
        retention_pct: Fraction of squad retained (0.0 to 1.0)
        base_decay: Minimum carry-forward fraction (default 0.3)
        baseline_elo: Neutral Elo to decay toward (default 1500)

    Returns:
        Adjusted Elo rating
    """
    # Carry-forward multiplier: ranges from base_decay (0% retention)
    # to 1.0 (100% retention)
    carry_forward = base_decay + retention_pct * (1.0 - base_decay)

    # Apply decay
    adjusted = baseline_elo + (old_elo - baseline_elo) * carry_forward
    return round(adjusted, 1)


def detect_season_boundary(prev_date, curr_date, matches: "pd.DataFrame" = None) -> bool:
    """
    Detect whether two consecutive match dates span a season boundary.
    IPL typically runs March-May; a gap of >90 days usually means a new season.
    """
    if prev_date is None:
        return False

    gap_days = (curr_date - prev_date).days
    return gap_days > 90


if __name__ == "__main__":
    print("Retention Decay module loaded.")
    print(f"\nExample: CSK 2024→2025 retention = {get_retention_pct('Chennai Super Kings', '2024', '2025'):.0%}")
    print(f"  Old Elo 1580 → Adjusted: {apply_retention_decay(1580, 0.50):.1f}")
    print(f"  Old Elo 1580 → Full rebuild (25%): {apply_retention_decay(1580, 0.25):.1f}")
    print(f"  Old Elo 1580 → Full retention (100%): {apply_retention_decay(1580, 1.0):.1f}")
