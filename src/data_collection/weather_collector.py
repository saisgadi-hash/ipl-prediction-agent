"""
Collect weather data for IPL match venues.

Uses the OpenWeatherMap API to get weather conditions (temperature,
humidity, wind, dew point) for each IPL venue.

HOW TO RUN:
    python -m src.data_collection.weather_collector

SETUP FIRST:
    1. Go to https://openweathermap.org/ and create a free account
    2. Copy your API key
    3. Create a .env file: cp .env.example .env
    4. Paste your API key: WEATHER_API_KEY=your_key_here

BEGINNER NOTES:
    - An API (Application Programming Interface) is like a waiter at a restaurant:
      you send a request (your order), and it brings back data (your food)
    - API keys are like passwords that identify your app to the service
    - JSON is the format most APIs use to send data back
    - We store the API key in .env file (never in code) for security
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import WEATHER_API_KEY, VENUES, EXTERNAL_DATA_DIR


def get_weather_for_city(city: str, lat: float, lon: float) -> dict:
    """
    Get current weather data for a city using OpenWeatherMap API.

    Args:
        city: City name (for display purposes)
        lat: Latitude of the venue
        lon: Longitude of the venue

    Returns:
        Dictionary with weather data, or empty dict if API call fails

    BEGINNER NOTE:
        - requests.get() sends an HTTP GET request to the API URL
        - params={} adds query parameters to the URL (like ?lat=18.93&lon=72.82)
        - response.json() converts the API response from JSON text to a Python dictionary
    """
    if not WEATHER_API_KEY:
        print(f"    SKIPPED {city}: No API key configured")
        return {}

    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": WEATHER_API_KEY,
        "units": "metric",  # Celsius, not Fahrenheit
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 401:
            print(f"    ERROR: Invalid API key. Check your .env file.")
            return {}
        elif response.status_code != 200:
            print(f"    ERROR: API returned status {response.status_code} for {city}")
            return {}

        data = response.json()

        weather = {
            "city": city,
            "temperature_c": data["main"]["temp"],
            "feels_like_c": data["main"]["feels_like"],
            "humidity_pct": data["main"]["humidity"],
            "pressure_hpa": data["main"]["pressure"],
            "wind_speed_mps": data["wind"]["speed"],
            "wind_direction_deg": data["wind"].get("deg", 0),
            "cloud_cover_pct": data["clouds"]["all"],
            "weather_condition": data["weather"][0]["main"],
            "weather_description": data["weather"][0]["description"],
            "visibility_m": data.get("visibility", 10000),
            "dew_point_c": calculate_dew_point(
                data["main"]["temp"], data["main"]["humidity"]
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add rain data if available
        if "rain" in data:
            weather["rain_1h_mm"] = data["rain"].get("1h", 0)
        else:
            weather["rain_1h_mm"] = 0

        return weather

    except requests.exceptions.ConnectionError:
        print(f"    ERROR: Could not connect to weather API for {city}")
        return {}
    except requests.exceptions.Timeout:
        print(f"    ERROR: Request timed out for {city}")
        return {}
    except Exception as e:
        print(f"    ERROR: {e}")
        return {}


def calculate_dew_point(temp_c: float, humidity_pct: float) -> float:
    """
    Calculate dew point temperature using the Magnus formula.

    Dew point is crucial for IPL predictions because:
    - When dew forms on the field, bowling becomes much harder
    - Teams batting second benefit hugely from dew
    - Dew is most likely when humidity > 60% and temp drops in evening

    BEGINNER NOTE:
        This is a physics formula. You don't need to understand the math.
        Just know that:
        - Higher humidity + lower temperature = more dew
        - Dew point close to air temperature = dew is very likely
    """
    import math

    a = 17.27
    b = 237.7
    alpha = (a * temp_c) / (b + temp_c) + math.log(humidity_pct / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return round(dew_point, 1)


def estimate_dew_probability(temp_c: float, humidity_pct: float, hour: int) -> float:
    """
    Estimate the probability of dew affecting the match.

    Args:
        temp_c: Current temperature in Celsius
        humidity_pct: Current relative humidity percentage
        hour: Hour of the day (0-23) in local time

    Returns:
        Probability between 0 and 1

    BEGINNER NOTE:
        Evening matches (after 7 PM) in India often have dew, especially
        when humidity is above 60%. This is a simplified estimation.
    """
    dew_point = calculate_dew_point(temp_c, humidity_pct)
    temp_dew_diff = temp_c - dew_point

    # Base probability from temperature-dew point difference
    if temp_dew_diff <= 2:
        base_prob = 0.9
    elif temp_dew_diff <= 5:
        base_prob = 0.6
    elif temp_dew_diff <= 10:
        base_prob = 0.3
    else:
        base_prob = 0.1

    # Evening matches have more dew
    if hour >= 19 or hour <= 5:  # 7 PM to 5 AM
        time_factor = 1.3
    elif hour >= 17:  # 5 PM to 7 PM
        time_factor = 1.1
    else:
        time_factor = 0.7

    return min(base_prob * time_factor, 1.0)


def collect_all_venue_weather():
    """
    Collect weather data for all IPL venues.

    Saves the data to data/external/venue_weather.csv
    """
    print("\n" + "#" * 60)
    print("#  COLLECTING WEATHER DATA FOR IPL VENUES")
    print("#" * 60)

    if not WEATHER_API_KEY:
        print("\n  WARNING: No weather API key found!")
        print("  To set up weather data:")
        print("    1. Go to https://openweathermap.org/api")
        print("    2. Sign up for a free account")
        print("    3. Copy your API key")
        print("    4. Run: cp .env.example .env")
        print("    5. Edit .env and paste your key")
        print("\n  Skipping weather collection for now.")
        print("  The model will work without weather data (just slightly less accurate).")
        return None

    weather_records = []

    for venue in VENUES:
        city = venue["city"]
        lat = venue["lat"]
        lon = venue["lon"]

        print(f"\n  Fetching weather for {city} ({venue['name']})...")
        weather = get_weather_for_city(city, lat, lon)

        if weather:
            # Add venue metadata
            weather["venue_name"] = venue["name"]
            weather["pitch_type"] = venue["pitch_type"]
            weather["dew_factor_rating"] = venue["dew_factor"]
            weather["dew_probability"] = estimate_dew_probability(
                weather["temperature_c"],
                weather["humidity_pct"],
                datetime.now().hour,
            )
            weather_records.append(weather)
            print(f"    Temp: {weather['temperature_c']}°C, "
                  f"Humidity: {weather['humidity_pct']}%, "
                  f"Dew prob: {weather['dew_probability']:.0%}")

        # Be respectful to the API: wait between requests
        time.sleep(1.5)

    if weather_records:
        df = pd.DataFrame(weather_records)
        output_path = EXTERNAL_DATA_DIR / "venue_weather.csv"
        df.to_csv(output_path, index=False)

        print("\n" + "=" * 60)
        print("WEATHER DATA COLLECTED!")
        print("=" * 60)
        print(f"  Venues: {len(df)}")
        print(f"  Saved to: {output_path}")
    else:
        print("\n  No weather data collected.")

    return weather_records


if __name__ == "__main__":
    collect_all_venue_weather()
