import pandas as pd

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

DATA_PATH = "data/census.csv"

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


def _load_small_df(n=300):
    return pd.read_csv(DATA_PATH).sample(n=n, random_state=42)


def test_process_data_returns_expected_shapes():
    df = _load_small_df(200)
    X, y, encoder, lb = process_data(
        df,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=True,
        encoder=None,
        lb=None,
    )
    assert X.shape[0] == df.shape[0]
    assert y.shape[0] == df.shape[0]
    assert encoder is not None
    assert lb is not None


def test_train_model_has_predict():
    df = _load_small_df(200)
    X, y, _, _ = process_data(
        df,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=True,
        encoder=None,
        lb=None,
    )
    model = train_model(X, y)
    assert hasattr(model, "predict")


def test_inference_output_length_matches_input():
    df = _load_small_df(200)
    X, y, _, _ = process_data(
        df,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=True,
        encoder=None,
        lb=None,
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y)


def test_compute_model_metrics_in_range():
    y = [0, 1, 1, 0, 1]
    preds = [0, 1, 0, 0, 1]
    precision, recall, f1 = compute_model_metrics(y, preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
