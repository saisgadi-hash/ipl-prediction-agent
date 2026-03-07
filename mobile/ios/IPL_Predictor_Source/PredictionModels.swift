import Foundation

// MARK: - API Request Request
struct MatchPredictionRequest: Codable {
    let team1: String
    let team2: String
    let venue: String?
}

// MARK: - API Response Models
struct MatchPredictionResponse: Codable {
    let predicted_winner: String
    let team1: String
    let team2: String
    let team1_win_probability: Double
    let team2_win_probability: Double
    let confidence: Double
    let justification: String?
    let llm_insight: String?
    let error: String?
}

enum LoadingState {
    case idle
    case loading
    case success(MatchPredictionResponse)
    case error(String)
}
