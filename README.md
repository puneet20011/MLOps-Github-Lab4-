# MLOps Lab 4 — FastAPI: Penguin Species Classifier API

Based on [`Labs/API_Labs/FastAPI_Labs`](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs) from the IE7374 MLOps course repo.
The original lab serves a Decision Tree trained on Iris. This version serves a different dataset and model, and adds more to the API.

## What I changed vs. the original lab

| | Original lab | This lab |
|---|---|---|
| **Dataset** | Iris (sklearn built-in, 4 numeric features) | **Palmer Penguins** CSV (`data/penguins.csv`): 4 numeric + 2 categorical features (`island`, `sex`), with missing rows dropped |
| **Model** | `DecisionTreeClassifier(max_depth=3)` | **sklearn `Pipeline`**: `StandardScaler` + `OneHotEncoder` → `RandomForestClassifier(200 trees)` |
| **Training output** | Model `.pkl` only | Model `.pkl` + `model_metadata.json` (classes, features, test accuracy & macro-F1, timestamp) |
| **Model loading** | Re-loaded from disk on **every** request | Loaded **once** at startup with FastAPI's `lifespan` handler |
| **Prediction response** | Class index (`0/1/2`) | Species **name**, confidence, and per-class probabilities |
| **Input validation** | Types only | `Field` range limits (e.g. no negative body mass) and `Literal` allowed values for `island` / `sex`, so bad input gets a **422** |
| **Endpoints** | `GET /`, `POST /predict` | `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch` (1–100 penguins) |
| **Error handling** | Generic 500 | **503** with a clear message if the model hasn't been trained |
| **Tests** | None | 10 pytest tests with FastAPI `TestClient` |
| **CI** | None | GitHub Actions: retrain + run tests on every push |

## Project structure

```
MLOps-FastAPI-Lab4
├── .github/workflows/test.yml   # CI: retrain + pytest
├── data/penguins.csv            # Palmer Penguins dataset (344 rows)
├── model/
│   ├── penguin_model.pkl        # trained pipeline
│   └── model_metadata.json      # metrics + feature list
├── src/
│   ├── data.py                  # load + stratified split
│   ├── train.py                 # build pipeline, train, save model + metadata
│   ├── predict.py               # PenguinModel wrapper (load once, predict_proba)
│   └── main.py                  # FastAPI app
├── tests/test_api.py
└── requirements.txt
```

## Setup & run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd src
python train.py                  # trains + prints metrics
uvicorn main:app --reload        # serve on http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

Run tests (from the project root):

```bash
pytest -v
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check + whether the model is loaded |
| GET | `/model-info` | Model type, classes, features, test accuracy / F1 |
| POST | `/predict` | Predict one penguin |
| POST | `/predict/batch` | Predict a list of 1–100 penguins |

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"bill_length_mm": 47.5, "bill_depth_mm": 15.0, "flipper_length_mm": 218,
       "body_mass_g": 4950, "island": "Biscoe", "sex": "female"}'
```

```json
{"species": "Gentoo", "confidence": 1.0,
 "probabilities": {"Adelie": 0.0, "Chinstrap": 0.0, "Gentoo": 1.0}}
```

Invalid input, such as `"island": "Hawaii"`, gets a 422 response:

```json
{"detail": [{"type": "literal_error", "loc": ["body", "island"],
  "msg": "Input should be 'Biscoe', 'Dream' or 'Torgersen'"}]}
```

## Results

On a stratified 80/20 split (266 train / 67 test), the model gets **1.00 test accuracy and 1.00 macro-F1**. Penguin species separate cleanly on bill and flipper measurements, so a perfect score on this dataset is common.

## Dataset credit

Horst AM, Hill AP, Gorman KB (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data.* https://allisonhorst.github.io/palmerpenguins/
