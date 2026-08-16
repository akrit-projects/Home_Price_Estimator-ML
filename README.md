# Home Price Estimator ML

A FastAPI machine learning API that predicts California home prices from neighborhood and location features.

## Production API

Live API documentation:

https://home-price-estimator-ml.onrender.com/docs

Base URL:

```text
https://home-price-estimator-ml.onrender.com
```

## API Paths

### `GET /`

Returns a basic API status message.

Example response:

```json
{
  "message": "California house price prediction api",
  "status": "running",
  "endpoint": "send  POST request to /predict"
}
```

### `GET /health`

Returns service health, model name, model features, and average error value configured in the API.

Example response:

```json
{
  "status": "running",
  "model": "RandomForestRegressor",
  "features": [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
  ],
  "avg_error": "$25,602"
}
```

### `POST /predict`

Predicts the estimated home price for one input record.

Request body:

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.984,
  "AveBedrms": 1.023,
  "Population": 322,
  "AveOccup": 2.555,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

Example curl request:

```bash
curl -X POST "https://home-price-estimator-ml.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.984,
    "AveBedrms": 1.023,
    "Population": 322,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

Example response:

```json
{
  "predicted_price": "$425,955",
  "predicted_price_short": "$4.26 hundred thouand",
  "fidence_range": "$400,353 to $451,557"
}
```

### `POST /predict-file`

Accepts a CSV file and returns a CSV file with predictions.

Required CSV columns:

```text
MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude
```

Example CSV:

```csv
MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude
8.3252,41,6.984,1.023,322,2.555,37.88,-122.23
2.5,30,4.5,1.2,1200,3.5,34.05,-118.25
```

Example curl request:

```bash
curl -X POST "https://home-price-estimator-ml.onrender.com/predict-file" \
  -F "file=@test_house.csv" \
  -o predictions.csv
```

The returned CSV includes an additional column:

```text
predicted_price_usd
```

## Model Details

Dataset:

```text
California Housing dataset from scikit-learn
```

Model:

```text
RandomForestRegressor
```

Training configuration:

```text
n_estimators=100
random_state=42
test_size=0.2
```

Target:

```text
Median house value in units of $100,000
```

The API converts model output to US dollars by multiplying predictions by `100000`.

## Features

| Feature | Meaning |
| --- | --- |
| `MedInc` | Median income in the block group |
| `HouseAge` | Median house age in the block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

## Model Performance

Measured on the same 20% test split used by `Models/House_predictor/train.py`.

| Metric | Value |
| --- | ---: |
| R2 score | `0.8044` |
| Mean Absolute Error | `$32,784` |
| Mean Squared Error | `0.2563` |
| Root Mean Squared Error | `$50,626` |

## Local Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API locally:

```bash
uvicorn main:app --reload
```

Local Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Deployment

The app is configured for Render with `render.yaml`.

Python version is set with `.python-version`.

Large model artifacts are tracked with Git LFS:

```text
Models/House_predictor/*.joblib
```
