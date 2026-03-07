import SwiftUI

struct ResultCard: View {
    let prediction: MatchPredictionResponse
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            
            // Winner Header
            HStack {
                VStack(alignment: .leading) {
                    Text("Predicted Winner")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .textCase(.uppercase)
                    
                    Text(prediction.predicted_winner)
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(Color.green)
                }
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("Confidence")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .textCase(.uppercase)
                    
                    Text("\(Int(prediction.confidence))%")
                        .font(.title2)
                        .fontWeight(.bold)
                }
            }
            .padding(.bottom, 8)
            
            Divider()
            
            // Win Probabilities
            VStack(alignment: .leading, spacing: 8) {
                Text("Win Probabilities")
                    .font(.headline)
                
                HStack {
                    Text(prediction.team1)
                    Spacer()
                    Text(String(format: "%.1f%%", prediction.team1_win_probability * 100))
                        .fontWeight(.bold)
                }
                
                // Probability Bar
                GeometryReader { geometry in
                    HStack(spacing: 0) {
                        Rectangle()
                            .fill(Color.blue)
                            .frame(width: geometry.size.width * CGFloat(prediction.team1_win_probability))
                        Rectangle()
                            .fill(Color.orange)
                            .frame(width: geometry.size.width * CGFloat(prediction.team2_win_probability))
                    }
                }
                .frame(height: 10)
                .cornerRadius(5)
                
                HStack {
                    Text(prediction.team2)
                    Spacer()
                    Text(String(format: "%.1f%%", prediction.team2_win_probability * 100))
                        .fontWeight(.bold)
                }
            }
            
            Divider()
            
            // Statistical Justification (SHAP)
            if let justification = prediction.justification {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Statistical Justification")
                        .font(.headline)
                    
                    Text(justification)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            
            // AI Tactical Preview (LLM)
            if let llmInsight = prediction.llm_insight, !llmInsight.isEmpty {
                Divider()
                
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: "sparkles")
                            .foregroundColor(.purple)
                        Text("AI Tactical Preview")
                            .font(.headline)
                    }
                    
                    Text(llmInsight)
                        .font(.subheadline)
                        .foregroundColor(.primary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding()
                .background(Color.purple.opacity(0.1))
                .cornerRadius(8)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
    }
}
