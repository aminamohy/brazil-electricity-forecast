from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os     
import joblib 
import pandas as pd
from datasets import load_dataset
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI(title="Brazil Electricity Forecast API", version="1.0.0")
security = HTTPBearer()

class ForecastRequest(BaseModel):
    region: Literal["Norte", "Nordeste", "Sudeste", "Sul"] = Field(..., description="Region of Brazil")
    prediction_period: Literal[24, 168] = Field(..., description="Prediction period")

class ForecastResponse(BaseModel):
    region: str
    prediction_period: int
    forecast: list[float]
    timestamps: list[str]

# =========== Load Models and Data (ONCE at startup) ============
print("🔄 Loading models and historical data..str.")

# Load historical data ONCE
dataset = load_dataset("SamuelM0422/Hourly-Electricity-Demand-Brazil-Dataset")
df = dataset['train'].to_pandas()
df.columns = ['region_id', 'region_name', 'date', 'total_load']
print(f"✅ Loaded historical data: {df.shape}")

# Prepare data for each region
historical_data = {}
regions = ['N', 'NE', 'SE', 'S']
for region in regions:
    region_df = df[df['region_id'] == region].drop(['region_name'], axis=1)
    historical_data[region] = region_df
    print(f"✅ Prepared data for region {region}: {region_df.shape}")

# Load models ONCE
models = {}
horizons = [24, 168]
for region in regions:
    for horizon in horizons:
        model_path = f'models/NBEATS_{region}_{horizon}h_model.pkl'
        try:
            models[f'{region}_{horizon}'] = joblib.load(model_path)
            print(f"✅ Loaded model: {model_path}")
        except FileNotFoundError:
            print(f"⚠️  Model not found: {model_path}")

mapping_region = {
    "Norte": "N",
    "Nordeste": "NE",
    "Sudeste": "SE",
    "Sul": "S"
}

print("🚀 API ready!")

#===========API health check endpoint================
@app.get("/")
async def root():
    """API health check endpoint"""
    return {
        "message": "Brazil Electricity Forecast API",
        "status": "running",
        "models_loaded": len(models),
        "regions": list(mapping_region.keys())
    }

#=========== Authentication Dependency ============
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return True

@app.post("/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest, authorized: bool = Depends(verify_api_key)):
    """
    Generate electricity load forecast for a specific region.
    
    - **region**: Region name (Norte, Nordeste, Sudeste, Sul)
    - **prediction_period**: Forecast horizon (24 or 168 hours)
    
    Returns forecasted values and timestamps.
    """
    region = request.region
    prediction_period = request.prediction_period

    # Get region code and model
    region_code = mapping_region.get(region)
    if not region_code:
        raise HTTPException(status_code=400, detail=f"Invalid region: {region}")
    
    model_key = f'{region_code}_{prediction_period}'
    model = models.get(model_key)
    
    if model is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Model not found for region {region} and period {prediction_period}h"
        )
    
    # Get historical data for this region
    region_data = historical_data.get(region_code)
    if region_data is None:
        raise HTTPException(status_code=500, detail="Historical data not available")
    
    try:
        # Make prediction using historical data
        forecast_df = model.predict(df=region_data)
        
        # Extract forecast values and timestamps
        forecast_list = forecast_df['NBEATS'].tolist()
        timestamps = forecast_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        
        return ForecastResponse(
            region=region,
            prediction_period=prediction_period,
            forecast=forecast_list,
            timestamps=timestamps
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


