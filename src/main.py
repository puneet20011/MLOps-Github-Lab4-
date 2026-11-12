"""FastAPI app that serves the Palmer Penguins species classifier."""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from predict import PenguinModel

models: dict[str, PenguinModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup instead of on every request.
    try:
        models["penguin"] = PenguinModel()
    except FileNotFoundError:
        pass  # /health will report it; /predict returns 503
    yield
    models.clear()


app = FastAPI(
    title="Penguin Species Classifier API",
    description="Predicts Adelie / Chinstrap / Gentoo from body measurements (Palmer Penguins).",
    version="1.0.0",
    lifespan=lifespan,
)


class PenguinFeatures(BaseModel):
    """One penguin. Field ranges reject physically impossible values (422)."""
    bill_length_mm: float = Field(..., gt=0, le=100, examples=[39.1])
    bill_depth_mm: float = Field(..., gt=0, le=40, examples=[18.7])
    flipper_length_mm: float = Field(..., gt=0, le=300, examples=[181])
    body_mass_g: float = Field(..., gt=0, le=10000, examples=[3750])
    island: Literal["Biscoe", "Dream", "Torgersen"] = Field(..., examples=["Torgersen"])
    sex: Literal["male", "female"] = Field(..., examples=["male"])


class PredictionResponse(BaseModel):
    species: str
    confidence: float
    probabilities: dict[str, float]


class BatchRequest(BaseModel):
    penguins: list[PenguinFeatures] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


def get_model() -> PenguinModel:
    model = models.get("penguin")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run `python train.py` first.",
        )
    return model


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {"status": "healthy", "model_loaded": "penguin" in models}


@app.get("/model-info")
async def model_info():
    """Metadata saved at training time: classes, features, test accuracy/F1."""
    return get_model().metadata


@app.post("/predict", response_model=PredictionResponse)
async def predict(penguin: PenguinFeatures):
    return get_model().predict([penguin.model_dump()])[0]


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchRequest):
    preds = get_model().predict([p.model_dump() for p in batch.penguins])
    return {"count": len(preds), "predictions": preds}
