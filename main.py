from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("Models\House_predictor\house_model.joblib")
features = joblib.load("Models\House_predictor\house_features.joblib")


class Housefeatures(BaseModel):
    MedInc: float = Field(gt = 0, description="Median income of Neighbourhood")
    HouseAge: float  = Field(gt = 0, description="Avg. House age in the block")
    AveRooms: float = Field(gt = 0, description="Avg. Number of rooms per house")
    AveBedrms: float = Field(gt = 0, description="Avg. Number of bedrooms per house")
    Population: float = Field(gt = 0, description="Total population of the block")
    AveOccup: float = Field(gt = 0, description="Avg. number of member per house")
    Latitude: float = Field(gt = 32, le=42, description="Latitude")
    Longitude: float = Field(gt = -125, le=-114, description="Longitude")


@app.get('/')
def home():
    return {
        "message": "California house price prediction api",
        "status" : "running",
        "endpoint": "send  POST request to /predict"
    }


@app.get('/health')
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$25,602"
    }

@app.post("/predict")
def predict(house: Housefeatures):
    try:
        input_data  = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude

        }])
        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000
        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"${predicted:,.2f} hundred thouand",
            "fidence_range": f"${price_usd - 25602:,.0f} to ${price_usd + 25602:,.0f}"
        }  
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction Failed: {str(e)}"
        )




