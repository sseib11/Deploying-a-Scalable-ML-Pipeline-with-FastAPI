from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import process_data
from ml.model import inference, load_model

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl")

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"

app = FastAPI(title="Census Income Classifier")


class CensusRecord(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        populate_by_name = True


def _load_artifacts() -> tuple[Any, Any, Any]:
    model = load_model(MODEL_PATH)
    encoder = load_model(ENCODER_PATH)
    lb = load_model(LB_PATH)
    return model, encoder, lb


MODEL, ENCODER, LB = _load_artifacts()


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Hello from the API!"}


@app.post("/predict")
def predict(record: CensusRecord) -> Dict[str, str]:
    row = record.model_dump(by_alias=True)
    df = pd.DataFrame([row])

    X, _, _, _ = process_data(
        df,
        categorical_features=CATEGORICAL_FEATURES,
        label=None,
        training=False,
        encoder=ENCODER,
        lb=LB,
    )

    pred = inference(MODEL, X)[0]
    label = LB.classes_[int(pred)]
    return {"prediction": str(label)}
