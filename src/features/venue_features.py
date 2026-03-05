"""
Venue and conditions-based feature engineering.

Every IPL venue has its own character: Chinnaswamy favours batsmen,
Chepauk helps spinners, Wankhede gets dew at night. This module
captures venue-specific patterns as features.

FEATURES GENERATED:
    - Average first/second innings score at venue
    - Toss-decision impact (bat vs field success rate)
    - Spin vs pace performance at venue
    - Dew factor and its impact on second innings
    - Home team advantage at venue
"""

import numpy as np
import pandas as pd


def calculate_venue_stats(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive venue statistics.

    Returns one row per venue with all relevant metrics.
    """
    print("  Calculating venue features...")

    venues = matches["venue"].unique()
    records = []

    for venue in venues:
        venue_matches = matches[matches["venue"] == venue]
        venue_deliveries = deliveries[deliveries["match_id"].isin(venue_matches["match_id"])]

        if len(venue_matches) < 3:
            continue

        record = {"venue": venue, "city": venue_matches["city"].mode().iloc[0] if len(venue_matches) > 0 else ""}

        # ── Scoring patterns ──
        innings1 = venue_deliveries[venue_deliveries["innings"] == 1]
        innings2 = venue_deliveries[venue_deliveries["innings"] == 2]

        inn1_scores = innings1.groupby("match_id")["runs_total"].sum()
        inn2_scores = innings2.groupby("match_id")["runs_total"].sum()

        record["venue_avg_1st_innings"] = round(inn1_scores.mean(), 1) if len(inn1_scores) > 0 else 160
        record["venue_avg_2nd_innings"] = round(inn2_scores.mean(), 1) if len(inn2_scores) > 0 else 155
        record["venue_avg_total"] = round(
            (record["venue_avg_1st_innings"] + record["venue_avg_2nd_innings"]) / 2, 1
        )
        record["venue_std_score"] = round(inn1_scores.std(), 1) if len(inn1_scores) > 1 else 20
        record["venue_matches_played"] = len(venue_matches)

        # ── High-scoring vs low-scoring ──
        all_innings_scores = pd.concat([inn1_scores, inn2_scores])
        record["venue_200_plus_pct"] = round(
            (all_innings_scores > 200).mean() * 100, 1
        ) if len(all_innings_scores) > 0 else 0
        record["venue_sub_140_pct"] = round(
            (all_innings_scores < 140).mean() * 100, 1
        ) if len(all_innings_scores) > 0 else 0

        # ── Toss impact ──
        toss_winners = venue_matches[venue_matches["toss_winner"] == venue_matches["winner"]]
        record["venue_toss_win_match_pct"] = round(
            len(toss_winners) / max(len(venue_matches), 1) * 100, 1
        )

        # Bat-first vs field-first win rates
        bat_first = venue_matches[venue_matches["toss_decision"] == "bat"]
        bat_first_toss_wins = bat_first[bat_first["toss_winner"] == bat_first["winner"]]
        record["venue_bat_first_win_pct"] = round(
            len(bat_first_toss_wins) / max(len(bat_first), 1) * 100, 1
        )

        field_first = venue_matches[venue_matches["toss_decision"] == "field"]
        field_first_toss_wins = field_first[field_first["toss_winner"] == field_first["winner"]]
        record["venue_field_first_win_pct"] = round(
            len(field_first_toss_wins) / max(len(field_first), 1) * 100, 1
        )

        # ── Chasing record ──
        chasing_wins = venue_matches[venue_matches["win_by_wickets"] > 0]
        record["venue_chasing_win_pct"] = round(
            len(chasing_wins) / max(len(venue_matches), 1) * 100, 1
        )

        # ── Boundary rates ──
        total_balls = len(venue_deliveries)
        if total_balls > 0:
            fours = (venue_deliveries["runs_batter"] == 4).sum()
            sixes = (venue_deliveries["runs_batter"] == 6).sum()
            record["venue_boundary_pct"] = round((fours + sixes) / total_balls * 100, 2)
            record["venue_six_pct"] = round(sixes / total_balls * 100, 2)
        else:
            record["venue_boundary_pct"] = 0
            record["venue_six_pct"] = 0

        # ── Phase-wise scoring ──
        for phase in ["powerplay", "middle", "death"]:
            phase_data = venue_deliveries[venue_deliveries["phase"] == phase]
            if len(phase_data) > 0:
                phase_runs = phase_data.groupby("match_id")["runs_total"].sum()
                record[f"venue_{phase}_avg_runs"] = round(phase_runs.mean(), 1)
                record[f"venue_{phase}_avg_wickets"] = round(
                    phase_data.groupby("match_id")["is_wicket"].sum().mean(), 2
                )
            else:
                record[f"venue_{phase}_avg_runs"] = 0
                record[f"venue_{phase}_avg_wickets"] = 0

        records.append(record)

    df = pd.DataFrame(records)
    print(f"    Calculated features for {len(df)} venues")
    return df


def calculate_team_venue_record(
    matches: pd.DataFrame, team: str, venue: str
) -> dict:
    """Calculate a specific team's record at a specific venue."""
    team_venue = matches[
        ((matches["team1"] == team) | (matches["team2"] == team)) &
        (matches["venue"] == venue)
    ]

    total = len(team_venue)
    wins = len(team_venue[team_venue["winner"] == team])

    return {
        "team_venue_matches": total,
        "team_venue_wins": wins,
        "team_venue_win_pct": round(wins / max(total, 1), 4),
    }


def is_home_venue(team: str, venue: str, team_config: list) -> bool:
    """Check if a venue is the home ground for a team."""
    for team_info in team_config:
        if team_info["name"] == team or team_info["code"] == team:
            return venue == team_info.get("home_venue", "")
    return False
