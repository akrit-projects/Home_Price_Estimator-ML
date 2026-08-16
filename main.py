from pathlib import Path
import io

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import joblib
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI()

MODEL_DIR = Path(__file__).resolve().parent / "Models" / "House_predictor"
model = joblib.load(MODEL_DIR / "house_model.joblib")
features = joblib.load(MODEL_DIR / "house_features.joblib")


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




@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )
    contents = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV file: {str(e)}"
        )

    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"File is missing required features {missing_features}"
        )
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="File contains no rows"
        )
    try:
        predictions = model.predict(df[features])
        df["predicted_price_usd"] = [f"${price * 100000:,.0f}" for price in predictions]
        output = df.to_csv(index=False)
        return StreamingResponse(
            io.BytesIO(output.encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions.csv"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction Failed: {str(e)}"
        )
