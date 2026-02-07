"""
Train and evaluate the model, save artifacts, and write slice metrics.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

DATA_PATH = "data/census.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl")
SLICE_OUTPUT_PATH = "slice_output.txt"

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


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    train_data, test_data = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data[LABEL]
    )

    X_train, y_train, encoder, lb = process_data(
        train_data,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=True,
        encoder=None,
        lb=None,
    )

    X_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    save_model(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    save_model(encoder, ENCODER_PATH)
    print(f"Encoder saved to {ENCODER_PATH}")
    save_model(lb, LB_PATH)
    print(f"Label binarizer saved to {LB_PATH}")

    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    with open(SLICE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for feature in CATEGORICAL_FEATURES:
            slice_metrics = performance_on_categorical_slice(
                data=test_data,
                feature=feature,
                label=LABEL,
                categorical_features=CATEGORICAL_FEATURES,
                encoder=encoder,
                lb=lb,
                model=model,
            )

            for m in slice_metrics:
                f.write(
                    f"Precision: {m.precision:.4f} | Recall: {m.recall:.4f} "
                    f"| F1: {m.f1:.4f}\n"
                )
                f.write(f"{m.feature}: {m.value}, Count: {m.count}\n")

    print(f"Slice metrics written to {SLICE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
