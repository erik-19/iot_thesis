import paho.mqtt.client as mqtt
import json
import pandas as pd
from datetime import datetime, time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib


# Load the pre-trained model
model = joblib.load('rf_tl_008T.pkl')


required_fields = {"sensor", "eCO2", "temperature", "humidity", "color_g"}
accumulated_data = {}
predictions_log = pd.DataFrame()

# Define the order of the features as used during model training
feature_order = [
    'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 
    'Time_stamp', 'Week_status_1'
]

def preprocess_data(df):
    df = df.drop(columns=["sensor"])
    df = df.rename(columns={
        "eCO2": "CO2",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "color_g": "Light"
    })
    
    # Calculate HumidityRatio
    df["Temperature"] = df["Temperature"] / 100
    df["HumidityRatio"] = df.apply(lambda row: calculate_humidity_ratio(row["Temperature"], row["Humidity"]), axis=1)
    
    df["Time_stamp"] = df["date_time"].apply(second_day)
    df["Week_status"] = df["date_time"].apply(weekend_weekday)
    df["Week_status"] = df["Week_status"].apply(relevel_weekend)
    df = pd.get_dummies(df, columns=['Week_status'], drop_first=True)
    
    if 'Week_status_1' not in df.columns:
        df['Week_status_1'] = 1
    week_status_columns = [col for col in df.columns if col.startswith('Week_status_')]
    for col in week_status_columns:
        df[col] = df[col].astype('category')

    df = df.drop(columns="date_time")
    
    # Define feature ranges
    feature_ranges = {
        'Temperature': (19.0, 24.4083), 
        'Humidity': (16.745, 39.5), 
        'Light': (0.0, 1697.25),
        'CO2': (412.75, 2076.5),
        'HumidityRatio': (0.00267412, 0.00647601),
    }
    feature_ranges_thingy008 = {
        'Temperature': (24.2, 32.94), 
        'Humidity': (21, 49), 
        'Light': (0.0, 515),
        'CO2': (420, 7385),
        'HumidityRatio': (0.028277, 0.071152),
    }
    
    # Normalize using historical ranges
    for feature in df.columns:
        if feature in feature_ranges_thingy008:
            min_val, max_val = feature_ranges_thingy008[feature]
            df[feature] = (df[feature] - min_val) / (max_val - min_val)
    
    # Re-scale to target ranges
    for feature in df.columns:
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            df[feature] = df[feature] * (max_val - min_val) + min_val

    # Ensure the DataFrame has the correct column order
    df = df[feature_order]

    return df

def calculate_humidity_ratio(temperature, humidity):
    Psat = np.exp(13.8193 - 2696.04 / (temperature + 224.317))
    vapor_pressure = (humidity / 100) * Psat
    atmospheric_pressure = 101.325
    humidity_ratio = 0.622 * vapor_pressure / (atmospheric_pressure - vapor_pressure)
    return humidity_ratio

def second_day(x):
    return x.hour * 3600 + x.minute * 60 + x.second

def weekend_weekday(x):
    if x.weekday() == 5 or x.weekday() == 6:
        return "Weekend"
    else:
        return "Weekday"

def relevel_weekend(x):
    if x == "Weekend":
        return 0
    else:
        return 1

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("#") 

def on_message(client, userdata, msg):
    global accumulated_data, predictions_log
    try:
        payload = msg.payload.decode()
        message = json.loads(payload)
        if message.get("sensor") == "thingy008":
            print(f"Topic: {msg.topic}\nMessage: {payload}")
            for key, value in message.items():
                if key in required_fields:
                    accumulated_data[key] = value

            # Check if all required fields are present
            if required_fields.issubset(accumulated_data.keys()):
                # Add a "date_time" field for preprocessing purposes
                accumulated_data["date_time"] = datetime.now()
                df = pd.DataFrame([accumulated_data])
                processed_df = preprocess_data(df)
                print(f"Processed DataFrame Columns: {processed_df.columns.tolist()}")
                print(f"Processed DataFrame:\n{processed_df}")
                # Ensure all features are included
                missing_features = set(feature_order) - set(processed_df.columns)
                if missing_features:
                    print(f"Missing features: {missing_features}")
                prediction = model.predict(processed_df)
                print(f"Prediction: {prediction[0]}")
                processed_df['Prediction'] = prediction[0]
                processed_df['Timestamp'] = datetime.now()
                # Record the prediction
                predictions_log = pd.concat([predictions_log, processed_df], ignore_index=True)
                predictions_log.to_csv("predictions_log.csv", index=False)
                # Clear the accumulated data
                accumulated_data.clear()
    except json.JSONDecodeError:
        print(f"Failed to decode JSON message: {msg.payload}")
    except Exception as e:
        print(f"Error: {e}")

def on_disconnect(client, userdata, rc):
    print("Disconnected from MQTT broker")
    # Attempt to reconnect
    try:
        client.reconnect()
    except Exception as e:
        print(f"Failed to reconnect: {e}")
        time.sleep(5) 

# Define the server IP, port, username, and password
broker_ip = "130.37.53.49"
broker_port = 1883
username = "surfdemo"
password = "HbUbwO-KuBn9"


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

client.username_pw_set(username, password)
client.connect(broker_ip, broker_port, 60)

client.loop_forever()
