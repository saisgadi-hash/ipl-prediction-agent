"""
Team Name Mapper — Standardise all historical team names.

Over the years, several IPL teams have been renamed or rebranded:
    - Delhi Daredevils → Delhi Capitals (2019)
    - Kings XI Punjab → Punjab Kings (2021)
    - Royal Challengers Bangalore → Royal Challengers Bengaluru (2024)
    - Deccan Chargers → Sunrisers Hyderabad (2013, different ownership but same city)
    - Rising Pune Supergiant / Rising Pune Supergiants (2016-2017, temporary team)
    - Gujarat Lions (2016-2017, temporary) — NOT the same as Gujarat Titans (2022+)
    - Kochi Tuskers Kerala (2011 only)
    - Pune Warriors India (2011-2013)

This module provides a single source of truth for mapping old names to current names.

HOW TO USE:
    from src.data_collection.team_name_mapper import standardise_team_name, apply_team_mapping

    name = standardise_team_name("Delhi Daredevils")  # → "Delhi Capitals"
    df = apply_team_mapping(df, columns=["team1", "team2", "winner", "toss_winner"])
"""

# ══════════════════════════════════════════
# TEAM NAME MAPPING
# ══════════════════════════════════════════
# Maps every historical name to the CURRENT (2025/2026) team name.
# Defunct teams (no current equivalent) keep their original name.

TEAM_NAME_MAP = {
    # Delhi
    "Delhi Daredevils": "Delhi Capitals",
    "Delhi Capitals": "Delhi Capitals",

    # Punjab
    "Kings XI Punjab": "Punjab Kings",
    "Punjab Kings": "Punjab Kings",

    # Bengaluru / Bangalore
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",

    # These are the same across seasons (no rename)
    "Chennai Super Kings": "Chennai Super Kings",
    "Mumbai Indians": "Mumbai Indians",
    "Kolkata Knight Riders": "Kolkata Knight Riders",
    "Rajasthan Royals": "Rajasthan Royals",
    "Sunrisers Hyderabad": "Sunrisers Hyderabad",
    "Lucknow Super Giants": "Lucknow Super Giants",
    "Gujarat Titans": "Gujarat Titans",

    # Defunct teams — keep original name (they have no current equivalent)
    "Deccan Chargers": "Deccan Chargers",
    "Kochi Tuskers Kerala": "Kochi Tuskers Kerala",
    "Pune Warriors": "Pune Warriors",
    "Pune Warriors India": "Pune Warriors",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Rising Pune Supergiants": "Rising Pune Supergiant",  # Standardise the plural
    "Gujarat Lions": "Gujarat Lions",
}

# Short codes mapping
TEAM_SHORT_CODES = {
    "Chennai Super Kings": "CSK",
    "Mumbai Indians": "MI",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Delhi Capitals": "DC",
    "Punjab Kings": "PBKS",
    "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH",
    "Lucknow Super Giants": "LSG",
    "Gujarat Titans": "GT",
    "Deccan Chargers": "DCH",
    "Kochi Tuskers Kerala": "KTK",
    "Pune Warriors": "PWI",
    "Rising Pune Supergiant": "RPS",
    "Gujarat Lions": "GL",
}

# Current active teams (2025/2026 season)
ACTIVE_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Lucknow Super Giants",
    "Gujarat Titans",
]


def standardise_team_name(name: str) -> str:
    """
    Convert any historical team name to the current standardised name.

    Args:
        name: Team name (any historical variant)

    Returns:
        Standardised current team name

    Examples:
        >>> standardise_team_name("Delhi Daredevils")
        'Delhi Capitals'
        >>> standardise_team_name("Kings XI Punjab")
        'Punjab Kings'
        >>> standardise_team_name("Royal Challengers Bangalore")
        'Royal Challengers Bengaluru'
    """
    if not name or not isinstance(name, str):
        return name

    name = name.strip()

    # Direct lookup
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]

    # Case-insensitive lookup
    name_lower = name.lower()
    for old_name, new_name in TEAM_NAME_MAP.items():
        if old_name.lower() == name_lower:
            return new_name

    # If not found, return as-is (might be a new team or typo)
    return name


def get_short_code(name: str) -> str:
    """Get the short code for a team (e.g., 'CSK' for 'Chennai Super Kings')."""
    standardised = standardise_team_name(name)
    return TEAM_SHORT_CODES.get(standardised, standardised[:3].upper())


def apply_team_mapping(df, columns=None):
    """
    Apply team name standardisation to a DataFrame.

    Args:
        df: pandas DataFrame
        columns: List of column names to standardise.
                 If None, auto-detects team name columns.

    Returns:
        DataFrame with standardised team names

    Example:
        df = apply_team_mapping(df, columns=["team1", "team2", "winner", "toss_winner"])
    """
    import pandas as pd

    if columns is None:
        # Auto-detect columns that likely contain team names
        team_keywords = ["team", "winner", "toss_winner", "batting_team", "bowling_team", "loser"]
        columns = [col for col in df.columns if any(kw in col.lower() for kw in team_keywords)]

    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: standardise_team_name(x) if isinstance(x, str) else x
            )

    return df


def print_mapping_summary():
    """Print a human-readable summary of all name changes."""
    print("\nIPL Team Name Standardisation Map:")
    print("=" * 50)

    renames = {}
    for old, new in TEAM_NAME_MAP.items():
        if old != new:
            if new not in renames:
                renames[new] = []
            renames[new].append(old)

    for current, old_names in sorted(renames.items()):
        print(f"\n  {current} ({get_short_code(current)})")
        for old in old_names:
            print(f"    ← was: {old}")

    print(f"\n  Active teams (2025/26): {len(ACTIVE_TEAMS)}")
    print(f"  Total historical names mapped: {len(TEAM_NAME_MAP)}")


if __name__ == "__main__":
    print_mapping_summary()
