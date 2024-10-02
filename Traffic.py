import streamlit as st
import requests
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# API Keys
tomtom_api_key = 'n4fVCepK41RI91HsP52iXrTAupnGKfSO'
owm_api_key = '09ecf05acbee2e40a21651a63de5e502'

# Function to fetch weather data
def fetch_weather_data(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={owm_api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'temperature': data['main']['temp'],
        'precipitation': data.get('rain', {}).get('1h', 0),
        'weather_description': data['weather'][0]['description']
    }

# Function to fetch traffic data
def fetch_traffic_data(location):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={location}&key={tomtom_api_key}"
    response = requests.get(url)
    data = response.json()
    return {
        'current_speed': data['flowSegmentData']['currentSpeed'],
        'free_flow_speed': data['flowSegmentData']['freeFlowSpeed']
    }

# Function to load or create data storage with initial mock data
def load_data():
    if os.path.exists('traffic_weather_data.csv'):
        return pd.read_csv('traffic_weather_data.csv')
    else:
        # Creating initial mock data for testing purposes
        initial_data = {
            'current_speed': [15, 20, 12, 18, 30, 10, 25, 22, 15, 17],
            'free_flow_speed': [30, 30, 25, 35, 40, 20, 35, 30, 30, 25],
            'temperature': [28, 30, 25, 31, 29, 27, 33, 29, 26, 28],
            'precipitation': [0, 0.1, 0, 0.5, 0, 0, 0.2, 0, 0.3, 0.1]
        }
        return pd.DataFrame(initial_data)

# Function to save data
def save_data(df):
    df.to_csv('traffic_weather_data.csv', index=False)

# Main Streamlit app
st.title("Traffic and Weather Analysis")

# Input section
st.sidebar.header("Enter Location Coordinates")
latitude = st.sidebar.text_input("Latitude", "13.0827")
longitude = st.sidebar.text_input("Longitude", "80.2707")

if st.sidebar.button("Fetch Data"):
    try:
        latitude = float(latitude)
        longitude = float(longitude)
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            st.error("Please enter valid latitude and longitude values.")
            st.stop()
    except ValueError:
        st.error("Please enter numeric values for latitude and longitude.")
        st.stop()

    location = f"{latitude},{longitude}"

    # Fetch traffic and weather data
    traffic_data = fetch_traffic_data(location)
    weather_data = fetch_weather_data(latitude, longitude)

    if traffic_data and weather_data:
        # Load previous data or create new data with mock entries
        df = load_data()

        # Add new data to DataFrame using pd.concat()
        new_data = pd.DataFrame({
            'current_speed': [traffic_data['current_speed']],
            'free_flow_speed': [traffic_data['free_flow_speed']],
            'temperature': [weather_data['temperature']],
            'precipitation': [weather_data['precipitation']]
        })

        df = pd.concat([df, new_data], ignore_index=True)

        # Save updated data
        save_data(df)

        # Display fetched data
        st.subheader("Traffic Data")
        st.write(f"Current Speed: {traffic_data['current_speed']} km/h")
        st.write(f"Free Flow Speed: {traffic_data['free_flow_speed']} km/h")

        st.subheader("Weather Data")
        st.write(f"Temperature: {weather_data['temperature']}°C")
        st.write(f"Precipitation: {weather_data['precipitation']} mm")
        st.write(f"Weather Description: {weather_data['weather_description']}")

        # Display the map
        st.subheader("Location Map")
        st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))

        # Model Prediction (Simple Linear Regression)
        st.subheader("Predicted Traffic Speed")

        # Calculate speed ratio for prediction features
        df['speed_ratio'] = df['current_speed'] / df['free_flow_speed']

        # Check if there are enough data points for prediction
        if len(df) >= 5:  # Adjusted threshold for testing
            X = df[['speed_ratio', 'temperature', 'precipitation']]
            y = df['current_speed']
            
            # Train the model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make prediction using the latest data
            predicted_speed = model.predict([[new_data['current_speed'][0] / new_data['free_flow_speed'][0], 
                                               new_data['temperature'][0], 
                                               new_data['precipitation'][0]]])[0]

            # Display prediction and model evaluation metrics
            st.write(f"Predicted Speed: {predicted_speed:.2f} km/h")

            # Evaluate model performance
            y_pred = model.predict(X_test)
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

            # Visualize Data
            st.subheader("Traffic Data Trends")
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=df, x=df.index, y='current_speed', label='Current Speed')
            plt.title("Traffic Speed Over Time")
            plt.xlabel("Data Points")
            plt.ylabel("Speed (km/h)")
            st.pyplot(plt)
        else:
            st.write("Not enough data points to perform a prediction. Please accumulate more data.")
            st.write(f"Predicted Speed: {traffic_data['current_speed']} km/h")  # Show current speed as fallback

# Option to clear the data
if st.sidebar.button("Clear Data"):
    if os.path.exists('traffic_weather_data.csv'):
        os.remove('traffic_weather_data.csv')
        st.success("Data cleared successfully!")
    else:
        st.warning("No data file found.")
