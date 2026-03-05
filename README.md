# IPL Prediction AI Agent

An AI-powered prediction engine that forecasts IPL match outcomes and tournament winners using machine learning, historical data, player statistics, and real-time adaptation.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ipl-prediction-agent.git
cd ipl-prediction-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 5. Run the full pipeline (download data → train model)
python run_pipeline.py

# 6. Launch the dashboard
streamlit run src/dashboard/app.py
```

## Run Individual Steps

```bash
# Data Collection
python -m src.data_collection.download_cricsheet   # Download match data
python -m src.data_collection.parse_matches         # Parse into CSVs
python -m src.data_collection.build_player_stats    # Build player stats
python -m src.data_collection.weather_collector     # Collect weather data

# Feature Engineering
python -m src.features.build_match_features         # Build all features

# Model Training
python -m src.models.train_model                    # Train ensemble model
python -m src.models.backtest                       # Back-test on past seasons

# Predictions
python -m src.models.predict --team1 "CSK" --team2 "MI"
python -m src.models.predict --tournament           # Tournament rankings

# Testing
pytest tests/ -v                                     # Run all tests
python -m tests.test_data_quality                    # Data quality checks
```

## Project Structure

```
ipl-prediction-agent/
├── config/              # Configuration files
├── data/
│   ├── raw/             # Original downloaded data
│   ├── processed/       # Cleaned, feature-engineered data
│   └── external/        # Weather, news, external data
├── models/              # Trained model files
├── notebooks/           # Jupyter notebooks for exploration
├── src/
│   ├── data_collection/ # Data download & parsing scripts
│   ├── features/        # Feature engineering modules
│   ├── models/          # ML model training & prediction
│   └── dashboard/       # Streamlit web dashboard
├── tests/               # Test files
├── .github/workflows/   # CI/CD automation
├── requirements.txt     # Python dependencies
└── run_pipeline.py      # Master pipeline runner
```

## Tech Stack

- **Language:** Python 3.12+
- **ML:** XGBoost, LightGBM, scikit-learn, SHAP
- **Data:** pandas, NumPy
- **Dashboard:** Streamlit, Plotly
- **Data Sources:** Cricsheet.org, OpenWeatherMap API
- **CI/CD:** GitHub Actions
- **Deployment:** Streamlit Cloud (free)
