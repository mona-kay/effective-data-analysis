import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression # type: ignore
import requests
import datetime
import json
import re

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('rats_weather.csv')
    return df

# Prepare and train the model
def train_model(df):
    df["day"] = pd.to_datetime(df["day"])
    df["weekday"] = df["day"].dt.dayofweek < 5
    df["weekday"] = df["weekday"].astype(int)
    X = df[['low_temp', 'wind_speed', 'weekday']]
    y = df['rat_sightings']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Fetch weather forecast data
def get_forecast():
    url = "https://api.weather.gov/gridpoints/OKX/34,36/forecast"
    response = requests.get(url)
    forecast_data = response.json()

    forecast_df = pd.DataFrame(forecast_data["properties"]["periods"])
    forecast_df[
        ["endTime", "isDaytime", "temperature", 
    "relativeHumidity", "windSpeed"]
    ].head()

    def get_wind_speed(row):
        numbers = [int(num) for num in re.findall(r'\d+', row)]
        return max(numbers)

    forecast_df["date"] = pd.to_datetime(forecast_df['endTime']).dt.date

    forecast_df["wind_speed"] = forecast_df["windSpeed"].apply(
        get_wind_speed
    )

    forecast_df["humidity"] = forecast_df["relativeHumidity"].apply(
        lambda x: x["value"]
    )

    daily_forecast = (
        forecast_df.groupby("date")
        .agg({
            "temperature": ["min", "max"],
            "humidity": "max",
            "wind_speed": "max"})
        .reset_index()
    )

    daily_forecast.columns = [
    "date", "low_temp", "high_temp", "humidity", "wind_speed"
    ]
    
    daily_forecast["date"] = pd.to_datetime(daily_forecast["date"])
    daily_forecast["dow"] = daily_forecast["date"].dt.dayofweek
    daily_forecast["weekday"] = (daily_forecast["dow"]<5).astype(int)

    return daily_forecast

# Streamlit application
def main():
    st.title("Rat Sightings Prediction Model")

    data = load_data()
    model = train_model(data)

    st.write("### Predicted Sightings")

    forecast_data = get_forecast()
    df_forecast = pd.DataFrame(forecast_data, columns=['date', 'low_temp', 'wind_speed', 'weekday'])
    predictions = model.predict(df_forecast[['low_temp', 'wind_speed', 'weekday']])

    # Prepare data for plotting
    df_forecast['predictions'] = predictions
    df_forecast.set_index('date', inplace=True)
    
    # Plot predictions using Streamlit's native bar chart
    st.bar_chart(df_forecast['predictions'], height=200)

    st.write("### Weather Forecast")
    st.table(df_forecast[["low_temp", "wind_speed", "predictions"]])
if __name__ == "__main__":
    main()
