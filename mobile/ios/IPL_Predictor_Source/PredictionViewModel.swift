import Foundation
import Combine
import SwiftUI

@MainActor
class PredictionViewModel: ObservableObject {
    @Published var team1: String = "Chennai Super Kings"
    @Published var team2: String = "Mumbai Indians"
    @Published var state: LoadingState = .idle
    
    // Replace this with your actual Mac's IP address if testing on a real iPhone.
    // Use 127.0.0.1 (localhost) if testing on the Xcode Simulator.
    private let apiUrl = "http://127.0.0.1:8001/predict_match"
    
    let allTeams = [
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", 
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians", 
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru", 
        "Sunrisers Hyderabad"
    ]
    
    func fetchPrediction() {
        if team1 == team2 {
            self.state = .error("Please select two different teams.")
            return
        }
        
        self.state = .loading
        
        guard let url = URL(string: apiUrl) else {
            self.state = .error("Invalid API URL")
            return
        }
        
        let requestBody = MatchPredictionRequest(team1: team1, team2: team2, venue: nil)
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONEncoder().encode(requestBody)
        } catch {
            self.state = .error("Failed to encode request body")
            return
        }
        
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                guard let self = self else { return }
                
                if let error = error {
                    self.state = .error("Network error: \(error.localizedDescription)\n\nMake sure your FastAPI server is running via 'uvicorn src.api.main:app' on your Mac.")
                    return
                }
                
                guard let data = data else {
                    self.state = .error("No data received from server")
                    return
                }
                
                do {
                    let decodedResponse = try JSONDecoder().decode(MatchPredictionResponse.self, from: data)
                    
                    if let apiError = decodedResponse.error {
                        self.state = .error(apiError)
                    } else {
                        self.state = .success(decodedResponse)
                    }
                } catch {
                    self.state = .error("Failed to decode response: \(error.localizedDescription)")
                }
            }
        }.resume()
    }
}
