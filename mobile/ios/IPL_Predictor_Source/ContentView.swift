import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = PredictionViewModel()
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Team Selectors
                    VStack {
                        HStack {
                            Text("Team 1")
                                .fontWeight(.semibold)
                            Spacer()
                            Picker("Team 1", selection: $viewModel.team1) {
                                ForEach(viewModel.allTeams, id: \.self) { team in
                                    Text(team).tag(team)
                                }
                            }
                            .pickerStyle(MenuPickerStyle())
                        }
                        
                        Divider()
                        
                        HStack {
                            Text("Team 2")
                                .fontWeight(.semibold)
                            Spacer()
                            Picker("Team 2", selection: $viewModel.team2) {
                                ForEach(viewModel.allTeams, id: \.self) { team in
                                    Text(team).tag(team)
                                }
                            }
                            .pickerStyle(MenuPickerStyle())
                        }
                    }
                    .padding()
                    .background(Color(.systemBackground))
                    .cornerRadius(12)
                    .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
                    .padding(.horizontal)
                    
                    // Predict Button
                    Button(action: {
                        viewModel.fetchPrediction()
                    }) {
                        Text("Predict Match")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(12)
                    }
                    .padding(.horizontal)
                    .disabled({
                        if case .loading = viewModel.state { return true }
                        return false
                    }())
                    
                    // Status & Results
                    switch viewModel.state {
                    case .idle:
                        Text("Select teams and tap predict to see the AI analysis.")
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding()
                            
                    case .loading:
                        ProgressView("Analyzing historical data and running match simulations...")
                            .padding()
                            
                    case .error(let message):
                        VStack(spacing: 12) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.red)
                                .font(.system(size: 40))
                            Text("Error")
                                .font(.headline)
                            Text(message)
                                .multilineTextAlignment(.center)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color(.systemBackground))
                        .cornerRadius(12)
                        .padding(.horizontal)
                        
                    case .success(let prediction):
                        ResultCard(prediction: prediction)
                            .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("IPL AI Predictor")
            .background(Color(.systemGroupedBackground).edgesIgnoringSafeArea(.all))
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
