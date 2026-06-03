from fastapi import FastAPI
import pandas as pd
import joblib
import os

from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk API", version="1.0")

# -------------------------
# LOAD LOCAL MODEL ONLY
# -------------------------
MODEL_PATH = "models/final_model.pkl"
model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/")
def home():
    return {"message": "Credit Risk API running"}

# -------------------------
# PREDICT
# -------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    input_df = pd.DataFrame([request.model_dump()])

    print(input_df.columns.tolist())

    probability = model.predict_proba(input_df)[0][1]

    return PredictionResponse(
        risk_probability=float(probability),
        is_high_risk=int(probability >= 0.5)
    )