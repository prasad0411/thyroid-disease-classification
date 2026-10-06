"""
FastAPI prediction service for thyroid disease classification.

Usage: uvicorn api.predict:app --reload   (run from the repo root)
"""
import os
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_registry import explain_one, load_latest, predict_one

bundle = load_latest()


def _warm_explainer():
    """Build the SHAP explainer in the background so the first /explain is fast."""
    try:
        explain_one(bundle, {"TSH": 2.5, "T3": 1.8, "T4": 105.0, "T4U": 1.0, "age": 45.0, "sex": 0,
                             "on_thyroxine": 0, "on_antithyroid": 0, "sick": 0,
                             "query_hypothyroid": 0, "query_hyperthyroid": 0})
        print("explainer warm", flush=True)
    except Exception as exc:  # never block startup
        print(f"explainer warmup skipped: {exc}", flush=True)


threading.Thread(target=_warm_explainer, daemon=True).start()

app = FastAPI(title="Thyroid Disease Classifier API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

Flag = Literal[0, 1]


class PatientInput(BaseModel):
    TSH: float = Field(2.5, ge=0, le=500)
    T3: float = Field(1.8, ge=0, le=20)
    T4: float = Field(105.0, ge=0, le=500)
    T4U: float = Field(1.0, ge=0, le=5)
    age: float = Field(45.0, ge=0, le=120)
    sex: Flag = 0
    on_thyroxine: Flag = 0
    on_antithyroid: Flag = 0
    sick: Flag = 0
    query_hypothyroid: Flag = 0
    query_hyperthyroid: Flag = 0


class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    features_used: list[str]
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle.model_name, "version": bundle.timestamp}


@app.get("/model-info")
def model_info():
    return {
        "model": bundle.model_name,
        "version": bundle.timestamp,
        "dataset_size": bundle.dataset_size,
        "majority_baseline": bundle.baseline,
        "test_metrics": bundle.metrics,
        "features": bundle.features,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    try:
        out = predict_one(bundle, patient.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictionOutput(**out, features_used=bundle.features, model_version=bundle.timestamp)


class Contribution(BaseModel):
    feature: str
    value: float
    shap: float


class ExplanationOutput(PredictionOutput):
    base_value: float
    contributions: list[Contribution]
    method: str


@app.post("/explain", response_model=ExplanationOutput)
def explain(patient: PatientInput):
    try:
        out = explain_one(bundle, patient.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return ExplanationOutput(**out, features_used=bundle.features, model_version=bundle.timestamp)
