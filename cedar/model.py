from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from cedar.data import AMOUNT_COLUMNS, FEATURES, PAY_COLUMNS, sha256


def signed_log(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def candidates(seed: int):
    numeric = Pipeline([
        ("signed_log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    linear_features = ColumnTransformer([
        ("amounts", numeric, AMOUNT_COLUMNS),
        ("history", OneHotEncoder(handle_unknown="ignore", sparse_output=False), PAY_COLUMNS),
    ])
    tree_features = ColumnTransformer([
        ("amounts", "passthrough", AMOUNT_COLUMNS),
        ("history", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), PAY_COLUMNS),
    ])
    return {
        "prior": DummyClassifier(strategy="prior"),
        "logistic": Pipeline([
            ("features", linear_features),
            ("estimator", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("features", tree_features),
            ("estimator", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=180, max_leaf_nodes=15,
                min_samples_leaf=60, l2_regularization=8.0, early_stopping=False,
                categorical_features=[False] * len(AMOUNT_COLUMNS) + [True] * len(PAY_COLUMNS),
                random_state=seed,
            )),
        ]),
    }


def load_artifact(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "model.joblib"
    if sha256(path) != manifest["model_sha256"]:
        raise ValueError("Model checksum mismatch; refusing to load the artifact.")
    if manifest["versions"]["scikit-learn"] != sklearn.__version__:
        raise ValueError("Use the scikit-learn version pinned in requirements.lock.txt.")
    # joblib is executable serialization. Only load this project's locally trained artifact.
    model = joblib.load(path)
    if manifest["features"] != FEATURES:
        raise ValueError("Artifact feature contract differs from this service.")
    return model, manifest
