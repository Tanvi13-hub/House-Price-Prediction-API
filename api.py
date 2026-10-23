import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts house price per unit area using a trained Random Forest model.",
)

# Load the trained model (path relative to this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "real_estate_model_v2.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Same column names and order as used during training
FEATURES = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]


class HouseInput(BaseModel):
    transaction_date: float
    house_age: float
    distance_to_mrt: float
    convenience_stores: int
    latitude: float
    longitude: float


@app.get("/")
def root():
    return {"message": "House Price Prediction API is running. Open /docs to try it."}


@app.post("/predict")
def predict(data: HouseInput):
    values = [[
        data.transaction_date,
        data.house_age,
        data.distance_to_mrt,
        data.convenience_stores,
        data.latitude,
        data.longitude,
    ]]
    df = pd.DataFrame(values, columns=FEATURES)
    prediction = model.predict(df)
    return {"predicted_price_per_unit_area": round(float(prediction[0]), 2)}