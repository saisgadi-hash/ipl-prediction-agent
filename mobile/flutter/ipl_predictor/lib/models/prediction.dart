/// Data models for API responses.

class MatchPrediction {
  final String predictedWinner;
  final double winProbability;
  final double confidence;
  final String team1;
  final String team2;
  final String? justification;
  final List<TopFactor> topFactors;

  MatchPrediction({
    required this.predictedWinner,
    required this.winProbability,
    required this.confidence,
    required this.team1,
    required this.team2,
    this.justification,
    this.topFactors = const [],
  });

  factory MatchPrediction.fromJson(Map<String, dynamic> json) {
    List<TopFactor> factors = [];
    if (json['top_factors'] != null) {
      factors = (json['top_factors'] as List)
          .map((f) => TopFactor.fromJson(f))
          .toList();
    }
    return MatchPrediction(
      predictedWinner: json['predicted_winner'] ?? json['winner'] ?? 'Unknown',
      winProbability: (json['win_probability'] ?? json['probability'] ?? 0.5).toDouble(),
      confidence: (json['confidence'] ?? 0.5).toDouble(),
      team1: json['team1'] ?? '',
      team2: json['team2'] ?? '',
      justification: json['justification'],
      topFactors: factors,
    );
  }
}

class TopFactor {
  final String feature;
  final double impact;
  final String direction; // "positive" or "negative"
  final String description;

  TopFactor({
    required this.feature,
    required this.impact,
    required this.direction,
    required this.description,
  });

  factory TopFactor.fromJson(Map<String, dynamic> json) {
    return TopFactor(
      feature: json['feature'] ?? '',
      impact: (json['impact'] ?? json['shap_value'] ?? 0.0).toDouble(),
      direction: json['direction'] ?? ((json['impact'] ?? 0.0) >= 0 ? 'positive' : 'negative'),
      description: json['description'] ?? json['label'] ?? json['feature'] ?? '',
    );
  }
}

class TeamRanking {
  final String team;
  final double winProbability;
  final int rank;

  TeamRanking({required this.team, required this.winProbability, required this.rank});

  factory TeamRanking.fromJson(Map<String, dynamic> json, int rank) {
    return TeamRanking(
      team: json['team'] ?? '',
      winProbability: (json['probability'] ?? json['win_probability'] ?? 0.0).toDouble(),
      rank: rank,
    );
  }
}
