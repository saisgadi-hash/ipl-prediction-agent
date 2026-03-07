import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/prediction.dart';
import '../models/live_match.dart';

class ApiService {
  // Change this to your Railway API URL when deployed
  static String baseUrl = 'https://ipl-ai-prediction.up.railway.app';

  /// Fetch list of active IPL teams from API, with offline fallback.
  static Future<List<String>> getTeams() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/teams')).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return List<String>.from(data['teams']);
      }
    } catch (_) {}
    // Fallback if API is unreachable
    return [
      'Chennai Super Kings',
      'Delhi Capitals',
      'Gujarat Titans',
      'Kolkata Knight Riders',
      'Lucknow Super Giants',
      'Mumbai Indians',
      'Punjab Kings',
      'Rajasthan Royals',
      'Royal Challengers Bengaluru',
      'Sunrisers Hyderabad',
    ];
  }

  /// Get match prediction from API.
  static Future<MatchPrediction> predictMatch({
    required String team1,
    required String team2,
    String? venue,
    String? tossWinner,
    String? tossDecision,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/predict'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'team1': team1,
        'team2': team2,
        if (venue != null) 'venue': venue,
        if (tossWinner != null) 'toss_winner': tossWinner,
        if (tossDecision != null) 'toss_decision': tossDecision,
      }),
    ).timeout(const Duration(seconds: 30));

    if (response.statusCode == 200) {
      return MatchPrediction.fromJson(json.decode(response.body));
    } else {
      final error = json.decode(response.body);
      throw Exception(error['detail'] ?? 'Prediction failed');
    }
  }

  /// Get tournament rankings.
  static Future<List<TeamRanking>> getTournamentPredictions() async {
    final response = await http.get(
      Uri.parse('$baseUrl/tournament'),
    ).timeout(const Duration(seconds: 60));

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      final rankings = data['rankings'] as List;
      return rankings.asMap().entries.map((entry) {
        return TeamRanking.fromJson(entry.value as Map<String, dynamic>, entry.key + 1);
      }).toList();
    } else {
      throw Exception('Failed to load tournament predictions');
    }
  }

  /// Get live IPL matches (Phase D).
  static Future<List<LiveMatch>> getLiveMatches() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/live/matches'),
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final matches = data['matches'] as List;
        return matches.map((m) => LiveMatch.fromJson({'state': m})).toList();
      }
    } catch (_) {}
    return [];
  }

  /// Get live match details with prediction (Phase D).
  static Future<LiveMatch?> getLiveMatch(String matchId) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/live/match/$matchId'),
      ).timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        return LiveMatch.fromJson(json.decode(response.body));
      }
    } catch (_) {}
    return null;
  }
}
