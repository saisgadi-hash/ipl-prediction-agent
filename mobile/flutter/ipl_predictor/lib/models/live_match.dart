/// Data model for live match state from the API.

class LiveMatch {
  final String matchId;
  final String name;
  final String status;
  final String venue;
  final String team1;
  final String team2;
  final String team1Score;
  final String team2Score;
  final double team1RunRate;
  final double team2RunRate;
  final int innings;
  final int target;
  final double requiredRunRate;
  final bool matchStarted;
  final bool matchEnded;
  final String lastUpdated;

  // Live prediction (optional — only available during match)
  final double? liveProbTeam1;
  final double? liveProbTeam2;
  final String? predictedWinner;
  final String? momentum;
  final int? projectedScore;

  LiveMatch({
    required this.matchId,
    required this.name,
    required this.status,
    required this.venue,
    required this.team1,
    required this.team2,
    this.team1Score = '',
    this.team2Score = '',
    this.team1RunRate = 0,
    this.team2RunRate = 0,
    this.innings = 1,
    this.target = 0,
    this.requiredRunRate = 0,
    this.matchStarted = false,
    this.matchEnded = false,
    this.lastUpdated = '',
    this.liveProbTeam1,
    this.liveProbTeam2,
    this.predictedWinner,
    this.momentum,
    this.projectedScore,
  });

  factory LiveMatch.fromJson(Map<String, dynamic> json) {
    final state = json['state'] ?? json;
    final prediction = json['prediction'];

    return LiveMatch(
      matchId: state['match_id'] ?? '',
      name: state['name'] ?? '',
      status: state['status'] ?? '',
      venue: state['venue'] ?? '',
      team1: state['team1'] ?? '',
      team2: state['team2'] ?? '',
      team1Score: state['team1_score'] ?? '',
      team2Score: state['team2_score'] ?? '',
      team1RunRate: (state['team1_run_rate'] ?? 0).toDouble(),
      team2RunRate: (state['team2_run_rate'] ?? 0).toDouble(),
      innings: state['innings'] ?? 1,
      target: state['target'] ?? 0,
      requiredRunRate: (state['required_run_rate'] ?? 0).toDouble(),
      matchStarted: state['match_started'] ?? false,
      matchEnded: state['match_ended'] ?? false,
      lastUpdated: state['last_updated'] ?? '',
      liveProbTeam1: prediction != null ? (prediction['live_probability_team1'] ?? 0.5).toDouble() : null,
      liveProbTeam2: prediction != null ? (prediction['live_probability_team2'] ?? 0.5).toDouble() : null,
      predictedWinner: prediction?['predicted_winner'],
      momentum: prediction?['momentum'],
      projectedScore: prediction?['projected_score'],
    );
  }

  bool get isLive => matchStarted && !matchEnded;

  String get momentumIcon {
    switch (momentum) {
      case 'strong_positive': return '🚀';
      case 'positive': return '📈';
      case 'negative': return '📉';
      case 'strong_negative': return '💥';
      default: return '➡️';
    }
  }
}
