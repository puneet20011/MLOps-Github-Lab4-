"""Load the saved model once and make predictions."""
import json

import joblib
import pandas as pd

from train import METADATA_PATH, MODEL_PATH


class PenguinModel:
    def __init__(self):
        self.pipeline = joblib.load(MODEL_PATH)
        self.metadata = json.loads(METADATA_PATH.read_text())

    def predict(self, rows: list[dict]) -> list[dict]:
        """rows: list of feature dicts. Returns species + per-class probabilities."""
        df = pd.DataFrame(rows, columns=self.metadata["features"])
        probs = self.pipeline.predict_proba(df)
        classes = self.pipeline.classes_
        results = []
        for p in probs:
            best = int(p.argmax())
            results.append({
                "species": str(classes[best]),
                "confidence": round(float(p[best]), 4),
                "probabilities": {str(c): round(float(v), 4) for c, v in zip(classes, p)},
            })
        return results
