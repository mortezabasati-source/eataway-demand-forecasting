import pandas as pd
import numpy as np
import requests
import time
from datetime import date, timedelta
from pathlib import Path

# ============================================================================
# Config
# ============================================================================
OUTPUT_PATH = Path(__file__).parent / "weather_weekly.csv"

CITIES = {
    "Stockholm": {"lat": 59.3293, "lon": 18.0686},
    "Uppsala": {"lat": 59.8588, "lon": 17.6389},
    "Västerås": {"lat": 59.6162, "lon": 16.5528},
    "Gävle": {"lat": 60.6745, "lon": 17.1417},
}

# The Open-Meteo API endpoint for historical data is different from forecast.
# For simplicity, and because the free API limits historical forecast data,
# we use the 'archive' API for historical data and 'forecast' for the future.
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Historical: from 730 days ago up to yesterday
HISTORICAL_START = date.today() - timedelta(days=730)
HISTORICAL_END = date.today() - timedelta(days=1)

# Forecast: today up to 14 days in future
FORECAST_START = date.today()
FORECAST_END = date.today() + timedelta(days=14)

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "sunshine_duration"
]

def fetch_weather_segment(city_name, coords, url, start, end):
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": DAILY_VARS,
        "timezone": "Europe/Berlin"
    }
    
    # Simple retry mechanism
    for attempt in range(3):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            daily = data.get("daily", {})
            
            if not daily:
                return pd.DataFrame()
                
            df = pd.DataFrame({
                "date": pd.to_datetime(daily.get("time")),
                "temp_max": daily.get("temperature_2m_max"),
                "temp_min": daily.get("temperature_2m_min"),
                "temp_mean": daily.get("temperature_2m_mean"),
                "precip_total": daily.get("precipitation_sum"),
                "rain_total": daily.get("rain_sum"),
                "snow_total": daily.get("snowfall_sum"),
                "wind_max": daily.get("wind_speed_10m_max"),
                "gust_max": daily.get("wind_gusts_10m_max"),
                "sunshine_hrs": np.array(daily.get("sunshine_duration", [0]*len(daily.get("time")))) / 3600,
            })
            df["city"] = city_name
            return df
        except Exception as e:
            if attempt == 2:
                print(f"  ✗ Failed on {url} for {city_name}: {e}")
                return pd.DataFrame()
            time.sleep(2)

def fetch_weather_for_city(city_name, coords):
    """
    Fetch historical and forecast data separately, then combine.
    """
    print(f"Fetching weather for {city_name}...")
    
    df_hist = fetch_weather_segment(city_name, coords, ARCHIVE_URL, HISTORICAL_START, HISTORICAL_END)
    df_fore = fetch_weather_segment(city_name, coords, FORECAST_URL, FORECAST_START, FORECAST_END)
    
    if df_hist.empty and df_fore.empty:
        raise ValueError(f"Could not fetch any data for {city_name}")
        
    df = pd.concat([df_hist, df_fore], ignore_index=True)
    # Drop duplicates just in case dates overlap
    df = df.drop_duplicates(subset=["date"])
    return df

def aggregate_weekly(df):
    """
    Aggregate daily weather data into weekly ISO format
    """
    df["year"] = df["date"].dt.isocalendar().year
    df["week"] = df["date"].dt.isocalendar().week
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)
    
    # Calculate some daily flags before weekly aggregation
    df["is_rainy_day"] = (df["rain_total"] > 2.0).astype(int)
    df["is_snowy_day"] = (df["snow_total"] > 1.0).astype(int)
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    
    weekly = df.groupby(["city", "year_week"]).agg(
        temp_mean=("temp_mean", "mean"),
        temp_max=("temp_max", "max"),
        temp_min=("temp_min", "min"),
        temp_range=("temp_range", "mean"),
        precip_total=("precip_total", "sum"),
        rain_total=("rain_total", "sum"),
        snow_total=("snow_total", "sum"),
        wind_max=("wind_max", "max"),
        gust_max=("gust_max", "max"),
        sunshine_hrs=("sunshine_hrs", "sum"),
        rain_days=("is_rainy_day", "sum"),
        snow_days=("is_snowy_day", "sum"),
    ).reset_index()
    
    # Weekly flags
    weekly["is_rainy_week"] = (weekly["rain_days"] >= 3).astype(int)
    weekly["is_snowy_week"] = (weekly["snow_days"] >= 1).astype(int)
    weekly["is_cold"] = (weekly["temp_mean"] < 0).astype(int)
    weekly["is_hot"] = (weekly["temp_mean"] > 20).astype(int)
    
    return weekly

def main():
    print("=" * 70)
    print("Fetching weather data (Historical + Forecast)")
    print("=" * 70)
    
    all_dfs = []
    for city, coords in CITIES.items():
        try:
            city_df = fetch_weather_for_city(city, coords)
            all_dfs.append(city_df)
        except Exception as e:
            print(f"Failed to fetch weather for {city}: {e}")
            
    if not all_dfs:
        print("No weather data fetched. Relying on cache if available.")
        if OUTPUT_PATH.exists():
            print(f"Cache found at {OUTPUT_PATH}. Proceeding with old data.")
        else:
            print("No cache available. Creating an empty fallback file.")
            pd.DataFrame(columns=["city", "year_week"]).to_csv(OUTPUT_PATH, index=False)
        return
        
    combined_daily = pd.concat(all_dfs, ignore_index=True)
    weekly_weather = aggregate_weekly(combined_daily)
    
    weekly_weather.to_csv(OUTPUT_PATH, index=False)
    print(f"Weather data saved to {OUTPUT_PATH}")
    print(f"Total rows: {len(weekly_weather)}")
    print(f"Weeks covered: {weekly_weather['year_week'].min()} to {weekly_weather['year_week'].max()}")

if __name__ == "__main__":
    main()