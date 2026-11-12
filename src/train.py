"""Train a Random Forest pipeline on Palmer Penguins and save it + its metrics."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_data, split_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "penguin_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def build_pipeline() -> Pipeline:
    """Preprocessing (scale numbers, one-hot encode text) + classifier in one object,
    so the API can pass raw values straight in."""
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def fit_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metadata = {
        "model_type": "RandomForestClassifier",
        "dataset": "Palmer Penguins",
        "classes": list(pipeline.classes_),
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    print(json.dumps(fit_model(), indent=2))
