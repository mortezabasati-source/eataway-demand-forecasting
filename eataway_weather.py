"""
==============================================================================
Eataway Weather Data Fetching Script
==============================================================================
Fetches historical weather (Open-Meteo) + future forecast (SMHI / Open-Meteo)

Usage:  pip install requests pandas && python eataway_weather.py
Output: weather_weekly.csv (for training) + weather_forecast.csv (for prediction)
Requires internet connection!
==============================================================================
"""
import requests, pandas as pd, numpy as np, time, sys
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path("/Users/zihaoyang/Desktop/eataway")

CITY_COORDS = {
    "Stockholm": {"lat": 59.33, "lon": 18.07},
    "Uppsala":   {"lat": 59.86, "lon": 17.64},
    "Gävle":     {"lat": 60.67, "lon": 17.15},
    "Västerås":  {"lat": 59.61, "lon": 16.55},
}

ORT_TO_CITY = {
    "Gävle": "Gävle",
    "Stockholm Mellan": "Stockholm", "Stockholm N": "Stockholm",
    "Stockholm Södra": "Stockholm", "Stockholm Väst": "Stockholm",
    "Stockholm Xtra": "Stockholm", "Stockholm Östra": "Stockholm",
    "Uppsala Rutt": "Uppsala", "Västerås": "Västerås",
}


def fetch_history(city, lat, lon, start, end):
    """Open-Meteo historical weather (ERA5 reanalysis, free, no API key required)"""
    print(f"  📡 {city}: {start} → {end} ...", end=" ", flush=True)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
                 "precipitation_sum,rain_sum,snowfall_sum,"
                 "wind_speed_10m_max,wind_gusts_10m_max,sunshine_duration",
        "timezone": "Europe/Stockholm",
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame(d)
        df["date"] = pd.to_datetime(df["time"])
        df["city"] = city
        df = df.drop(columns=["time"])
        print(f"✓ {len(df)} days")
        return df
    except Exception as e:
        print(f"✗ {e}")
        return pd.DataFrame()


def fetch_forecast(city, lat, lon):
    """Open-Meteo 16-day forecast"""
    print(f"  🔮 {city} ...", end=" ", flush=True)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
                 "precipitation_sum,rain_sum,snowfall_sum,"
                 "wind_speed_10m_max,wind_gusts_10m_max,sunshine_duration,"
                 "precipitation_probability_max",
        "timezone": "Europe/Stockholm", "forecast_days": 16,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame(d)
        df["date"] = pd.to_datetime(df["time"])
        df["city"] = city
        df = df.drop(columns=["time"])
        print(f"✓ {len(df)} days")
        return df
    except Exception as e:
        print(f"✗ {e}")
        return pd.DataFrame()


def daily_to_weekly(history):
    """Aggregate daily data → weekly"""
    history["year"] = history["date"].dt.isocalendar().year.astype(int)
    history["week"] = history["date"].dt.isocalendar().week.astype(int)
    history["year_week"] = (history["year"].astype(str) + "-W" +
                            history["week"].astype(str).str.zfill(2))

    weekly = history.groupby(["city", "year", "week", "year_week"]).agg(
        temp_mean=("temperature_2m_mean", "mean"),
        temp_max=("temperature_2m_max", "max"),
        temp_min=("temperature_2m_min", "min"),
        precip_total=("precipitation_sum", "sum"),
        rain_total=("rain_sum", "sum"),
        snow_total=("snowfall_sum", "sum"),
        wind_max=("wind_speed_10m_max", "max"),
        gust_max=("wind_gusts_10m_max", "max"),
        sunshine_hrs=("sunshine_duration", lambda x: x.sum() / 3600),
        rain_days=("precipitation_sum", lambda x: (x > 1.0).sum()),
        snow_days=("snowfall_sum", lambda x: (x > 0).sum()),
    ).reset_index()

    weekly["temp_range"] = weekly["temp_max"] - weekly["temp_min"]
    weekly["is_rainy_week"] = (weekly["rain_days"] >= 3).astype(int)
    weekly["is_snowy_week"] = (weekly["snow_days"] >= 1).astype(int)
    weekly["is_cold"] = (weekly["temp_mean"] < 0).astype(int)
    weekly["is_hot"] = (weekly["temp_mean"] > 20).astype(int)
    return weekly


def main():
    print("╔" + "═" * 55 + "╗")
    print("║" + "  Eataway Weather Data Fetching".center(47) + "║")
    print("╚" + "═" * 55 + "╝\n")

    start = "2024-07-01"
    end = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

    # Historical
    print("1. Historical weather")
    frames = []
    for city, info in CITY_COORDS.items():
        df = fetch_history(city, info["lat"], info["lon"], start, end)
        if not df.empty: frames.append(df)
        time.sleep(1)

    if frames:
        hist = pd.concat(frames, ignore_index=True)
        weekly = daily_to_weekly(hist)
        p1 = f"{OUTPUT_DIR}/weather_weekly.csv"
        weekly.to_csv(p1, index=False, encoding="utf-8-sig")
        print(f"  ✓ {p1} ({len(weekly)} rows)\n")

    # Forecast
    print("2. Weather forecast")
    fframes = []
    for city, info in CITY_COORDS.items():
        df = fetch_forecast(city, info["lat"], info["lon"])
        if not df.empty: fframes.append(df)
        time.sleep(1)

    if fframes:
        fc = pd.concat(fframes, ignore_index=True)
        p2 = f"{OUTPUT_DIR}/weather_forecast.csv"
        fc.to_csv(p2, index=False, encoding="utf-8-sig")
        print(f"  ✓ {p2} ({len(fc)} rows)\n")

    # Mapping table
    mapping = pd.DataFrame([{"ort": k, "weather_city": v} for k, v in ORT_TO_CITY.items()])
    p3 = f"{OUTPUT_DIR}/ort_city_mapping.csv"
    mapping.to_csv(p3, index=False, encoding="utf-8-sig")
    print(f"  ✓ Mapping table: {p3}")
    print("\nDone! Next, run eataway_train_v4.py")


if __name__ == "__main__":
    main()