"""Load and split the Palmer Penguins dataset."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "penguins.csv"

NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "species"


def load_data(path: Path = DATA_PATH):
    """Read the CSV, drop rows with missing values, return (X, y)."""
    df = pd.read_csv(path).dropna(subset=FEATURES + [TARGET])
    return df[FEATURES], df[TARGET]


def split_data(X, y):
    """80/20 split, stratified so every species appears in both halves."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
