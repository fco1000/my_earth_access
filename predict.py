import joblib
import pandas as pd
from datetime import datetime

# === Load model ===
model = joblib.load("ndvi_predictor.pkl")

def interpret_ndvi(value):
    """
    Translate NDVI numeric value into human-friendly meaning.
    """
    if value < 0.1:
        return "Bare land or urban area — no vegetation."
    elif value < 0.3:
        return "Sparse vegetation — likely dry grass or scrubland."
    elif value < 0.5:
        return "Moderate vegetation — healthy shrubs or seasonal crops."
    elif value < 0.7:
        return "Dense vegetation — lush and green, normal bloom."
    else:
        return "Very dense vegetation — potential invasive or algal bloom!"

def is_anomaly(value, threshold=0.75):
    """
    Flag as anomaly if NDVI exceeds threshold.
    """
    return value > threshold

def predict_ndvi(lat, lon, date_str):
    """
    Predict NDVI and return interpretation.
    """
    ts = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
    df = pd.DataFrame([[lat, lon, ts]], columns=["lat", "lon", "timestamp"])
    ndvi = model.predict(df)[0]

    meaning = interpret_ndvi(ndvi)
    anomaly_flag = is_anomaly(ndvi)

    return {
        "latitude": lat,
        "longitude": lon,
        "date": date_str,
        "predicted_ndvi": float(ndvi),
        "interpretation": meaning,
        "anomaly": anomaly_flag
    }
# === Terminal interface ===
city = input("Enter City and date for NDVI prediction. \n Available cities are: \n 1. Kisumu\n 2. Nairobi \n 3. Kericho \n").strip().title()
cities = {
    "Kisumu":[0.0917,34.7680],
    "Nairobi":[1.2921, 36.8219],
    "Kericho":[0.3689, 35.2863]
}



if city in cities:
    lat,lon = cities[city]

date_str = input("Date (YYYY-MM-DD): ").strip()

result = predict_ndvi(lat, lon, date_str)

print("\nNDVI Prediction Result")
print(f"Location: {city} - ({result['latitude']}, {result['longitude']})")
print(f"Date: {result['date']}")
print(f"Predicted NDVI: {result['predicted_ndvi']:.4f}")
print(f"Meaning: {result['interpretation']}")
if result['anomaly']:
    print("Anomaly Detected: Possible invasive bloom or algal growth!")
else:
    print("Normal vegetation conditions.")
