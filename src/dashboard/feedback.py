"""
Anonymous User Feedback System with Gamification.

Allows users to:
1. Submit their own prediction (agree/disagree with the AI)
2. Provide reasoning for their prediction
3. Rate the AI's justification quality
4. Earn points and badges for participation

All feedback is ANONYMOUS — no login required, no personal data collected.
Data is stored locally in a JSON file.

HOW TO USE (in dashboard):
    from src.dashboard.feedback import FeedbackManager
    fm = FeedbackManager()
    fm.submit_prediction_feedback(match_id, user_pick, reasoning, ai_correct)

BEGINNER NOTES:
    - JSON files are a simple way to store structured data
    - UUID gives each submission a unique ID without needing user accounts
    - Gamification (points, badges, streaks) keeps users engaged
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "..", "data", "feedback"))


class FeedbackManager:
    """Manages anonymous user feedback with gamification."""

    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.data_dir / "user_feedback.json"
        self.leaderboard_file = self.data_dir / "leaderboard.json"
        self._load()

    def _load(self):
        """Load feedback data from disk."""
        if self.feedback_file.exists():
            with open(self.feedback_file, "r") as f:
                self.feedback = json.load(f)
        else:
            self.feedback = {"predictions": [], "ratings": [], "stats": {}}

        if self.leaderboard_file.exists():
            with open(self.leaderboard_file, "r") as f:
                self.leaderboard = json.load(f)
        else:
            self.leaderboard = {"users": {}}

    def _save(self):
        """Save feedback data to disk."""
        with open(self.feedback_file, "w") as f:
            json.dump(self.feedback, f, indent=2, default=str)
        with open(self.leaderboard_file, "w") as f:
            json.dump(self.leaderboard, f, indent=2, default=str)

    # ══════════════════════════════════════════
    # FEEDBACK SUBMISSION
    # ══════════════════════════════════════════

    def submit_prediction_feedback(
        self,
        team1: str,
        team2: str,
        ai_prediction: str,
        user_prediction: str,
        user_reasoning: str = "",
        session_id: str = None,
    ) -> dict:
        """
        Submit user's own prediction for a match (agree or disagree with AI).

        Args:
            team1: First team name
            team2: Second team name
            ai_prediction: Which team the AI predicted
            user_prediction: Which team the user predicts
            user_reasoning: Optional text explanation
            session_id: Anonymous session identifier (auto-generated if not provided)

        Returns:
            Dictionary with submission confirmation and points earned
        """
        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        entry = {
            "id": str(uuid.uuid4())[:12],
            "timestamp": datetime.now().isoformat(),
            "team1": team1,
            "team2": team2,
            "ai_prediction": ai_prediction,
            "user_prediction": user_prediction,
            "user_agrees_with_ai": user_prediction == ai_prediction,
            "user_reasoning": user_reasoning,
            "session_id": session_id,
            "actual_winner": None,  # Filled in later when match result is known
            "user_correct": None,   # Filled in later
            "ai_correct": None,     # Filled in later
        }

        self.feedback["predictions"].append(entry)

        # Award points
        points = self._award_points(session_id, "prediction", user_reasoning)

        self._save()

        return {
            "status": "submitted",
            "entry_id": entry["id"],
            "points_earned": points,
            "total_points": self.get_user_points(session_id),
            "badges": self.get_user_badges(session_id),
        }

    def submit_justification_rating(
        self,
        team1: str,
        team2: str,
        rating: int,
        comment: str = "",
        session_id: str = None,
    ) -> dict:
        """
        Rate the quality of the AI's prediction justification (1-5 stars).

        Args:
            team1: First team name
            team2: Second team name
            rating: 1-5 star rating
            comment: Optional feedback comment
            session_id: Anonymous session identifier
        """
        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        rating = max(1, min(5, rating))  # Clamp to 1-5

        entry = {
            "id": str(uuid.uuid4())[:12],
            "timestamp": datetime.now().isoformat(),
            "team1": team1,
            "team2": team2,
            "rating": rating,
            "comment": comment,
            "session_id": session_id,
        }

        self.feedback["ratings"].append(entry)
        points = self._award_points(session_id, "rating", comment)
        self._save()

        return {
            "status": "rated",
            "points_earned": points,
            "total_points": self.get_user_points(session_id),
        }

    def record_actual_result(self, team1: str, team2: str, actual_winner: str):
        """
        Record the actual match result and update user/AI accuracy.

        Call this after a match has been played to see who was right.
        """
        updated = 0
        for pred in self.feedback["predictions"]:
            if pred["team1"] == team1 and pred["team2"] == team2 and pred["actual_winner"] is None:
                pred["actual_winner"] = actual_winner
                pred["user_correct"] = pred["user_prediction"] == actual_winner
                pred["ai_correct"] = pred["ai_prediction"] == actual_winner

                # Bonus points for correct prediction
                if pred["user_correct"]:
                    self._award_points(pred["session_id"], "correct_prediction", "")

                # Extra bonus if user was right and AI was wrong
                if pred["user_correct"] and not pred["ai_correct"]:
                    self._award_points(pred["session_id"], "beat_ai", "")

                updated += 1

        self._save()
        return {"matches_updated": updated}

    # ══════════════════════════════════════════
    # GAMIFICATION — POINTS & BADGES
    # ══════════════════════════════════════════

    POINTS_TABLE = {
        "prediction": 10,          # Submit a prediction
        "prediction_with_reason": 15,  # With reasoning = bonus
        "rating": 5,               # Rate justification quality
        "rating_with_comment": 10, # With comment = bonus
        "correct_prediction": 25,  # Prediction was correct
        "beat_ai": 50,            # User right, AI wrong
    }

    BADGES = {
        "first_prediction": {"name": "Rookie Predictor", "icon": "🏏", "threshold": 1, "type": "predictions"},
        "five_predictions": {"name": "Regular Analyst", "icon": "📊", "threshold": 5, "type": "predictions"},
        "twenty_predictions": {"name": "Cricket Pundit", "icon": "🎙️", "threshold": 20, "type": "predictions"},
        "first_correct": {"name": "Sharp Eye", "icon": "🎯", "threshold": 1, "type": "correct"},
        "five_correct": {"name": "Form Reader", "icon": "🔮", "threshold": 5, "type": "correct"},
        "beat_ai_once": {"name": "AI Challenger", "icon": "🤖", "threshold": 1, "type": "beat_ai"},
        "beat_ai_five": {"name": "AI Slayer", "icon": "⚔️", "threshold": 5, "type": "beat_ai"},
        "streak_three": {"name": "Hot Streak", "icon": "🔥", "threshold": 3, "type": "streak"},
        "hundred_points": {"name": "Century Scorer", "icon": "💯", "threshold": 100, "type": "points"},
        "five_hundred_points": {"name": "IPL Expert", "icon": "🏆", "threshold": 500, "type": "points"},
    }

    def _award_points(self, session_id: str, action: str, text: str) -> int:
        """Award points for an action and return points earned."""
        if session_id not in self.leaderboard["users"]:
            self.leaderboard["users"][session_id] = {
                "points": 0,
                "predictions": 0,
                "correct": 0,
                "beat_ai": 0,
                "streak": 0,
                "best_streak": 0,
                "badges": [],
                "joined": datetime.now().isoformat(),
            }

        user = self.leaderboard["users"][session_id]

        # Determine points
        if action == "prediction":
            points = self.POINTS_TABLE["prediction_with_reason"] if text else self.POINTS_TABLE["prediction"]
            user["predictions"] += 1
        elif action == "rating":
            points = self.POINTS_TABLE["rating_with_comment"] if text else self.POINTS_TABLE["rating"]
        elif action == "correct_prediction":
            points = self.POINTS_TABLE["correct_prediction"]
            user["correct"] += 1
            user["streak"] += 1
            user["best_streak"] = max(user["best_streak"], user["streak"])
        elif action == "beat_ai":
            points = self.POINTS_TABLE["beat_ai"]
            user["beat_ai"] += 1
        else:
            points = 0

        user["points"] += points
        self._check_badges(session_id)
        return points

    def _check_badges(self, session_id: str):
        """Check and award any newly earned badges."""
        user = self.leaderboard["users"][session_id]
        earned = set(user.get("badges", []))

        for badge_id, badge in self.BADGES.items():
            if badge_id in earned:
                continue

            if badge["type"] == "predictions" and user["predictions"] >= badge["threshold"]:
                earned.add(badge_id)
            elif badge["type"] == "correct" and user["correct"] >= badge["threshold"]:
                earned.add(badge_id)
            elif badge["type"] == "beat_ai" and user["beat_ai"] >= badge["threshold"]:
                earned.add(badge_id)
            elif badge["type"] == "streak" and user["best_streak"] >= badge["threshold"]:
                earned.add(badge_id)
            elif badge["type"] == "points" and user["points"] >= badge["threshold"]:
                earned.add(badge_id)

        user["badges"] = list(earned)

    def get_user_points(self, session_id: str) -> int:
        """Get total points for a user."""
        return self.leaderboard["users"].get(session_id, {}).get("points", 0)

    def get_user_badges(self, session_id: str) -> list:
        """Get list of earned badges for a user."""
        badge_ids = self.leaderboard["users"].get(session_id, {}).get("badges", [])
        return [
            {**self.BADGES[bid], "id": bid}
            for bid in badge_ids if bid in self.BADGES
        ]

    def get_user_stats(self, session_id: str) -> dict:
        """Get complete stats for a user session."""
        user = self.leaderboard["users"].get(session_id, {})
        if not user:
            return {"points": 0, "predictions": 0, "correct": 0, "accuracy": 0, "badges": []}

        return {
            "points": user.get("points", 0),
            "predictions": user.get("predictions", 0),
            "correct": user.get("correct", 0),
            "accuracy": round(user["correct"] / max(user["predictions"], 1) * 100, 1),
            "beat_ai": user.get("beat_ai", 0),
            "streak": user.get("streak", 0),
            "best_streak": user.get("best_streak", 0),
            "badges": self.get_user_badges(session_id),
        }

    # ══════════════════════════════════════════
    # ANALYTICS — AGGREGATE STATS
    # ══════════════════════════════════════════

    def get_community_stats(self) -> dict:
        """Get aggregate community feedback statistics."""
        preds = self.feedback["predictions"]
        ratings = self.feedback["ratings"]

        total_preds = len(preds)
        resolved = [p for p in preds if p["actual_winner"] is not None]
        ai_correct = sum(1 for p in resolved if p["ai_correct"])
        user_correct = sum(1 for p in resolved if p["user_correct"])
        agreements = sum(1 for p in preds if p["user_agrees_with_ai"])

        return {
            "total_predictions": total_preds,
            "total_ratings": len(ratings),
            "avg_rating": round(sum(r["rating"] for r in ratings) / max(len(ratings), 1), 1),
            "ai_accuracy": round(ai_correct / max(len(resolved), 1) * 100, 1),
            "crowd_accuracy": round(user_correct / max(len(resolved), 1) * 100, 1),
            "agreement_rate": round(agreements / max(total_preds, 1) * 100, 1),
            "unique_sessions": len(self.leaderboard["users"]),
            "resolved_matches": len(resolved),
        }

    def get_leaderboard(self, top_n: int = 10) -> list:
        """Get the top users by points."""
        users = self.leaderboard.get("users", {})
        sorted_users = sorted(users.items(), key=lambda x: x[1].get("points", 0), reverse=True)

        return [
            {
                "rank": i + 1,
                "session_id": sid[:4] + "****",  # Partially masked for anonymity
                "points": data.get("points", 0),
                "predictions": data.get("predictions", 0),
                "correct": data.get("correct", 0),
                "accuracy": round(data["correct"] / max(data["predictions"], 1) * 100, 1),
                "badges_count": len(data.get("badges", [])),
            }
            for i, (sid, data) in enumerate(sorted_users[:top_n])
        ]

    def get_ai_vs_crowd_comparison(self) -> dict:
        """Compare AI vs crowd accuracy per team."""
        resolved = [p for p in self.feedback["predictions"] if p["actual_winner"] is not None]

        team_stats = {}
        for p in resolved:
            for team in [p["team1"], p["team2"]]:
                if team not in team_stats:
                    team_stats[team] = {"ai_correct": 0, "crowd_correct": 0, "total": 0}
                team_stats[team]["total"] += 1
                if p["ai_correct"]:
                    team_stats[team]["ai_correct"] += 1
                if p["user_correct"]:
                    team_stats[team]["crowd_correct"] += 1

        return {
            team: {
                "ai_accuracy": round(s["ai_correct"] / max(s["total"], 1) * 100, 1),
                "crowd_accuracy": round(s["crowd_correct"] / max(s["total"], 1) * 100, 1),
                "total_predictions": s["total"],
            }
            for team, s in team_stats.items()
        }
