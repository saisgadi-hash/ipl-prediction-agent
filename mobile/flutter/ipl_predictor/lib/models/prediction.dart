/// Data models for API responses.

/// Normalise a value to 0-1 range (handles both 0.65 and 65.0 formats)
double _normalise(dynamic val) {
  final v = (val ?? 0.5).toDouble();
  return v > 1.0 ? v / 100.0 : v;
}

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
      winProbability: _normalise(json['win_probability'] ?? json['probability'] ?? 0.5),
      confidence: _normalise(json['confidence'] ?? 0.5),
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
  final double eloRating;
  final String formState;
  // Phase E: Confidence intervals
  final double ciLower;
  final double ciUpper;
  // Phase E: Pythagorean Win Expectation
  final double pweExpected;
  final double pweActual;
  final double pweDiff;

  TeamRanking({
    required this.team,
    required this.winProbability,
    required this.rank,
    this.eloRating = 1500.0,
    this.formState = "Normal",
    this.ciLower = 0.0,
    this.ciUpper = 0.0,
    this.pweExpected = 0.5,
    this.pweActual = 0.5,
    this.pweDiff = 0.0,
  });

  factory TeamRanking.fromJson(Map<String, dynamic> json, int rank) {
    final prob = (json['probability'] ?? json['win_probability'] ?? 0.0).toDouble();
    return TeamRanking(
      team: json['team'] ?? '',
      winProbability: prob,
      rank: rank,
      eloRating: (json['elo_rating'] ?? 1500.0).toDouble(),
      formState: json['form_state'] ?? 'Normal',
      ciLower: (json['ci_lower'] ?? prob).toDouble(),
      ciUpper: (json['ci_upper'] ?? prob).toDouble(),
      pweExpected: (json['pwe_expected'] ?? 0.5).toDouble(),
      pweActual: (json['pwe_actual'] ?? 0.5).toDouble(),
      pweDiff: (json['pwe_diff'] ?? 0.0).toDouble(),
    );
  }
}
