from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
from typing import List
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load model and scaler
try:
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

app = FastAPI(
    title="House Price Prediction API",
    description="ML model PI for predicting house prices",
    version="1.0.0",
)


class HouseFeatures(BaseModel):
    square_feet: float = Field(..., gt=0, description="Square footage of the house")
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms")
    age: int = Field(..., ge=0, le=100, description="Age of the house in years")
    location_score: float = Field(
        ..., ge=0, le=10, description="Location quality score"
    )

    class Config:

        schema_extra = {
            "example": {
                "square_feet": 1500.0,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "age": 10,
                "location_score": 7.5,
            }
        }


class PredictionResponse(BaseModel):
    predicted_price: float
    prediction_range: dict


@app.get("/health")
async def health_check():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """Predict house price based on features"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")

    try:
        # prepare input
        input_data = np.array(
            [
                [
                    features.square_feet,
                    features.bedrooms,
                    features.bathrooms,
                    features.age,
                    features.location_score,
                ]
            ]
        )

        # scale features
        input_scaled = scaler.transform(input_data)

        # make prediction
        prediction = model.predict(input_scaled)[0]

        # calculate prediction range (+-10%)
        lower_bound = prediction * 0.9
        upper_bound = prediction * 1.1
        logger.info(f"Prediction: {prediction}, Range: ({lower_bound}, {upper_bound})")
        return PredictionResponse(
            predicted_price=prediction,
            prediction_range={"lower_bound": lower_bound, "upper_bound": upper_bound},
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")


@app.post("/predict/batch")
async def predict_batch(features_list: List[HouseFeatures]):
    """Make predictions for multiple houses"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")

    predictions = []
    for features in features_list:
        input_data = np.array(
            [
                [
                    features.square_feet,
                    features.bedrooms,
                    features.bathrooms,
                    features.age,
                    features.location_score,
                ]
            ]
        )
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        predictions.append({"input": features.dict(), "predicted_price": float(prediction)})
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
