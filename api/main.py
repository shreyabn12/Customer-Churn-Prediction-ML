# api/main.py
"""
FastAPI REST endpoint for customer churn prediction.

Endpoints:
  GET  /          - Health check
  GET  /info      - Model information
  POST /predict   - Single customer churn prediction
  POST /predict/batch - Multiple customers at once
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import uvicorn

# ── Load model once at startup ───────────────────────────────────────────────
# Loading happens once when the server starts, not on every request
# This makes predictions fast — no disk I/O per request

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')

try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"[API] Model loaded successfully from {MODEL_PATH}")
    model_name = type(pipeline.named_steps['classifier']).__name__
    print(f"[API] Model type: {model_name}")
except FileNotFoundError:
    print(f"[API] ERROR: Model not found at {MODEL_PATH}")
    print("[API] Please run src/train.py first to generate the model file")
    pipeline = None
    model_name = "Not loaded"


# ── FastAPI app setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    Predicts whether a telecom customer is likely to churn based on their 
    account information and usage patterns.
    
    Built with XGBoost + Optuna hyperparameter tuning + SMOTE class balancing.
    AUC-ROC: 0.8535 on held-out test set.
    """,
    version="1.0.0"
)

# Allow all origins for development
# In production you would restrict this to your frontend's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request and Response schemas ─────────────────────────────────────────────
# Pydantic models define exactly what data the API accepts and returns
# FastAPI uses these to validate incoming requests automatically
# If a required field is missing or wrong type, FastAPI returns a 422 error

class CustomerData(BaseModel):
    """
    Input schema — one customer's features.
    Each field has a description that appears in the auto-generated /docs page.
    Default values match the most common value in the dataset.
    """
    # Numeric fields
    tenure_months: int = Field(
        ..., ge=0, le=120,
        description="Number of months the customer has been with the company",
        example=12
    )
    monthly_charges: float = Field(
        ..., ge=0,
        description="Current monthly charge amount in dollars",
        example=65.5
    )
    total_charges: float = Field(
        ..., ge=0,
        description="Total amount charged over the customer's lifetime",
        example=786.0
    )

    # Service fields
    contract: str = Field(
        ...,
        description="Contract type",
        example="Month-to-month"
    )
    internet_service: str = Field(
        ...,
        description="Internet service provider type",
        example="Fiber optic"
    )
    payment_method: str = Field(
        ...,
        description="Payment method",
        example="Electronic check"
    )
    multiple_lines: str = Field(
        ...,
        description="Whether customer has multiple lines",
        example="No"
    )
    online_security: str = Field(
        ...,
        description="Whether customer has online security add-on",
        example="No"
    )
    online_backup: str = Field(
        ...,
        description="Whether customer has online backup add-on",
        example="No"
    )
    device_protection: str = Field(
        ...,
        description="Whether customer has device protection add-on",
        example="No"
    )
    tech_support: str = Field(
        ...,
        description="Whether customer has tech support add-on",
        example="No"
    )
    streaming_tv: str = Field(
        ...,
        description="Whether customer streams TV",
        example="Yes"
    )
    streaming_movies: str = Field(
        ...,
        description="Whether customer streams movies",
        example="Yes"
    )

    # Demographic fields
    gender: str = Field(
        ...,
        description="Customer gender: Male or Female",
        example="Male"
    )
    senior_citizen: int = Field(
        ..., ge=0, le=1,
        description="Whether the customer is a senior citizen: 0 or 1",
        example=0
    )
    partner: str = Field(
        ...,
        description="Whether the customer has a partner: Yes or No",
        example="No"
    )
    dependents: str = Field(
        ...,
        description="Whether the customer has dependents: Yes or No",
        example="No"
    )
    phone_service: str = Field(
        ...,
        description="Whether the customer has phone service: Yes or No",
        example="Yes"
    )
    paperless_billing: str = Field(
        ...,
        description="Whether the customer uses paperless billing: Yes or No",
        example="Yes"
    )


class PredictionResponse(BaseModel):
    """Output schema — what the API returns for each prediction."""
    churn_prediction: str           # "Yes" or "No"
    churn_probability: float        # probability of churn (0.0 to 1.0)
    risk_level: str                 # "High", "Medium", or "Low"
    confidence: str                 # human-readable confidence description
    model_used: str                 # name of the model that made the prediction


class BatchRequest(BaseModel):
    """Input schema for batch predictions — list of customers."""
    customers: List[CustomerData]


class BatchResponse(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    churn_rate_predicted: float


# ── Helper functions ─────────────────────────────────────────────────────────

def customer_to_dataframe(customer: CustomerData) -> pd.DataFrame:
    """
    Converts a Pydantic CustomerData object into a pandas DataFrame
    that matches exactly what our preprocessing pipeline expects.
    
    The column names must match what prepare_X_y() produces in features.py
    """
    data = {
        'gender': [customer.gender],
        'senior_citizen': [customer.senior_citizen],
        'partner': [customer.partner],
        'dependents': [customer.dependents],
        'tenure_months': [customer.tenure_months],
        'phone_service': [customer.phone_service],
        'multiple_lines': [customer.multiple_lines],
        'internet_service': [customer.internet_service],
        'online_security': [customer.online_security],
        'online_backup': [customer.online_backup],
        'device_protection': [customer.device_protection],
        'tech_support': [customer.tech_support],
        'streaming_tv': [customer.streaming_tv],
        'streaming_movies': [customer.streaming_movies],
        'contract': [customer.contract],
        'paperless_billing': [customer.paperless_billing],
        'payment_method': [customer.payment_method],
        'monthly_charges': [customer.monthly_charges],
        'total_charges': [customer.total_charges],
    }
    return pd.DataFrame(data)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same feature engineering we did during training.
    This MUST match exactly what features.py does — otherwise the model
    gets different inputs than it was trained on and predictions are wrong.
    """
    # charges_per_month
    df['charges_per_month'] = df.apply(
        lambda row: row['monthly_charges'] if row['tenure_months'] == 0
        else round(row['total_charges'] / row['tenure_months'], 2),
        axis=1
    )

    # tenure_group_encoded
    def get_tenure_group(tenure):
        if tenure <= 12:   return 0  # new
        elif tenure <= 24: return 1  # developing
        elif tenure <= 48: return 2  # established
        else:              return 3  # loyal

    df['tenure_group_encoded'] = df['tenure_months'].apply(get_tenure_group)

    # Encode binary columns — must match encode_binary_columns() in features.py
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(binary_map)

    return df


def get_risk_level(probability: float) -> tuple:
    """
    Converts a raw churn probability into a human-readable risk level
    and confidence description. Thresholds are based on business context.
    """
    if probability >= 0.7:
        return "High", f"Model is {probability:.0%} confident this customer will churn. Immediate retention action recommended."
    elif probability >= 0.4:
        return "Medium", f"Moderate churn risk ({probability:.0%}). Consider proactive outreach."
    else:
        return "Low", f"Low churn risk ({probability:.0%}). Customer appears stable."


def run_prediction(customer: CustomerData) -> PredictionResponse:
    """Runs the full prediction pipeline for a single customer."""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/train.py first."
        )

    # Convert to DataFrame
    df = customer_to_dataframe(customer)

    # Apply feature engineering (must match training)
    df = apply_feature_engineering(df)

    # Get prediction and probability
    prob = pipeline.predict_proba(df)[0][1]   # probability of churn (class 1)
    pred = "Yes" if prob >= 0.5 else "No"
    risk_level, confidence = get_risk_level(prob)

    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=round(float(prob), 4),
        risk_level=risk_level,
        confidence=confidence,
        model_used=model_name
    )


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint — confirms the API is running."""
    return {
        "status": "running",
        "message": "Customer Churn Prediction API is live",
        "model": model_name,
        "docs": "/docs"
    }


@app.get("/info", tags=["Health"])
def model_info():
    """Returns information about the loaded model."""
    return {
        "model_type": model_name,
        "auc_roc": 0.8535,
        "training_samples": 7043,
        "features": 21,
        "smote_applied": True,
        "hyperparameter_tuning": "Optuna (30 trials)",
        "dataset": "IBM Telco Customer Churn"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerData):
    """
    Predicts churn probability for a single customer.
    
    Send a POST request with customer data in JSON format.
    Returns churn prediction, probability, and risk level.
    """
    return run_prediction(customer)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """
    Predicts churn for multiple customers in one request.
    More efficient than calling /predict individually for each customer.
    """
    predictions = [run_prediction(c) for c in request.customers]

    high_risk = sum(1 for p in predictions if p.risk_level == "High")
    churn_predicted = sum(1 for p in predictions if p.churn_prediction == "Yes")

    return BatchResponse(
        predictions=predictions,
        total_customers=len(predictions),
        high_risk_count=high_risk,
        churn_rate_predicted=round(churn_predicted / len(predictions), 3)
    )


# ── Run server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True   # auto-restarts when you edit the file
    )