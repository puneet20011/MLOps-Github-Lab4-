import pytest
from fastapi.testclient import TestClient

from main import app

ADELIE = {"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181,
          "body_mass_g": 3750, "island": "Torgersen", "sex": "male"}
GENTOO = {"bill_length_mm": 47.5, "bill_depth_mm": 15.0, "flipper_length_mm": 218,
          "body_mass_g": 4950, "island": "Biscoe", "sex": "female"}
CHINSTRAP = {"bill_length_mm": 49.6, "bill_depth_mm": 18.2, "flipper_length_mm": 193,
             "body_mass_g": 3775, "island": "Dream", "sex": "male"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` runs the lifespan (loads the model)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["classes"] == ["Adelie", "Chinstrap", "Gentoo"]
    assert info["test_accuracy"] > 0.9


@pytest.mark.parametrize("payload,expected", [
    (ADELIE, "Adelie"), (GENTOO, "Gentoo"), (CHINSTRAP, "Chinstrap"),
])
def test_predict_species(client, payload, expected):
    body = client.post("/predict", json=payload).json()
    assert body["species"] == expected
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3


def test_batch(client):
    r = client.post("/predict/batch", json={"penguins": [ADELIE, GENTOO, CHINSTRAP]})
    assert r.status_code == 200
    assert [p["species"] for p in r.json()["predictions"]] == ["Adelie", "Gentoo", "Chinstrap"]


@pytest.mark.parametrize("bad", [
    {**ADELIE, "island": "Hawaii"},     # not a valid island
    {**ADELIE, "body_mass_g": -5},      # negative weight
    {k: v for k, v in ADELIE.items() if k != "sex"},  # missing field
])
def test_invalid_input_returns_422(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"penguins": []}).status_code == 422
