"""
Model utilities for training, inference, persistence, and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a Logistic Regression model."""
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    return model


def inference(model: Any, X: np.ndarray) -> np.ndarray:
    """Run model inference and return predictions."""
    return model.predict(X)


def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> tuple[float, float, float]:
    """Compute precision, recall, and F1."""
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = fbeta_score(y, preds, beta=1, zero_division=0)
    return precision, recall, f1


def save_model(obj: Any, path: str) -> None:
    """Save a model/encoder/label binarizer to disk."""
    joblib.dump(obj, path)


def load_model(path: str) -> Any:
    """Load a model/encoder/label binarizer from disk."""
    return joblib.load(path)


@dataclass
class SliceMetric:
    feature: str
    value: str
    count: int
    precision: float
    recall: float
    f1: float


def performance_on_categorical_slice(
    data: pd.DataFrame,
    feature: str,
    label: str,
    categorical_features: List[str],
    encoder: Any,
    lb: Any,
    model: Any,
) -> List[SliceMetric]:
    """
    Compute performance metrics on slices for each unique value of a categorical feature.
    """
    if feature not in data.columns:
        raise ValueError(f"Feature '{feature}' not found in data columns.")

    results: List[SliceMetric] = []

    for value in sorted(data[feature].dropna().unique()):
        df_slice = data[data[feature] == value].copy()
        count = df_slice.shape[0]
        if count == 0:
            continue

        X_slice, y_slice, _, _ = process_data(
            df_slice,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(model, X_slice)
        precision, recall, f1 = compute_model_metrics(y_slice, preds)

        results.append(
            SliceMetric(
                feature=feature,
                value=str(value),
                count=int(count),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
            )
        )

    return results
