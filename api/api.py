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
    description="Predict vegetation index (NDVI) for a given county and date.",
    version="1.0"
)

# === county coordinates ===
counties = {
    "Nairobi": [-1.2864, 36.8172],
    "Mombasa": [-4.0435, 39.6682],
    "Kisumu": [-0.0917, 34.7680],
    "Nakuru": [-0.3031, 36.0800],
    "Eldoret": [0.5143, 35.2698],
    "Machakos": [-1.5177, 37.2634],
    "Kilifi": [-3.6333, 39.8500],
    "Meru": [0.0472, 37.6500],
    "Nyeri": [-0.4167, 36.9500],
    "Garissa": [-0.4569, 39.6583],
    "Bungoma": [0.5635, 34.5606],
    "Narok": [-1.0800, 35.8700],
    "Voi": [-3.3961, 38.5561],
    "Nanyuki": [0.0167, 37.0667],
    "Homa Bay": [-0.5273, 34.4571]
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
    county: str
    date: str

class NDVIResponse(BaseModel):
    county: str
    latitude: float
    longitude: float
    date: str
    predicted_ndvi: float
    interpretation: str
    anomaly: bool

# === Main prediction route ===
@app.post("/predict", response_model=NDVIResponse)
def predict_ndvi(request: NDVIRequest):
    county = request.county.strip().title()

    if county not in counties:
        raise HTTPException(status_code=404, detail="county not supported. Choose Kisumu, Nairobi, or Kericho.")

    try:
        ts = int(datetime.strptime(request.date, "%Y-%m-%d").timestamp())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    lat, lon = counties[county]
    df = pd.DataFrame([[lat, lon, ts]], columns=["lat", "lon", "timestamp"])
    ndvi = model.predict(df)[0]

    return NDVIResponse(
        county=county,
        latitude=lat,
        longitude=lon,
        date=request.date,
        predicted_ndvi=float(ndvi),
        interpretation=interpret_ndvi(ndvi),
        anomaly=is_anomaly(ndvi)
    )
