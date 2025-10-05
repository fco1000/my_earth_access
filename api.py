from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

# === Load model ===
model = joblib.load("ndvi_predictor.pkl")

# === FastAPI setup ===
app = FastAPI(
    title="NDVI Prediction API",
    description="Predict vegetation index (NDVI) for a given city and date.",
    version="1.0"
)

# === City coordinates ===
CITIES = {
    "Kisumu": [0.0917, 34.7680],
    "Nairobi": [1.2921, 36.8219],
    "Kericho": [0.3689, 35.2863]
}

# === Helper functions ===
def interpret_ndvi(value):
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
    return value > threshold

# === Request and response models ===
class NDVIRequest(BaseModel):
    city: str
    date: str

class NDVIResponse(BaseModel):
    city: str
    latitude: float
    longitude: float
    date: str
    predicted_ndvi: float
    interpretation: str
    anomaly: bool

# === Main prediction route ===
@app.post("/predict", response_model=NDVIResponse)
def predict_ndvi(request: NDVIRequest):
    city = request.city.strip().title()

    if city not in CITIES:
        raise HTTPException(status_code=404, detail="City not supported. Choose Kisumu, Nairobi, or Kericho.")

    try:
        ts = int(datetime.strptime(request.date, "%Y-%m-%d").timestamp())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    lat, lon = CITIES[city]
    df = pd.DataFrame([[lat, lon, ts]], columns=["lat", "lon", "timestamp"])
    ndvi = model.predict(df)[0]

    return NDVIResponse(
        city=city,
        latitude=lat,
        longitude=lon,
        date=request.date,
        predicted_ndvi=float(ndvi),
        interpretation=interpret_ndvi(ndvi),
        anomaly=is_anomaly(ndvi)
    )
