import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cedar.api import ScoreRequest, create_app
from cedar.data import FEATURES, PAY_COLUMNS, TARGET, make_splits
from cedar.model import load_artifact

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def payload():
    return json.loads((ROOT / "examples/request.json").read_text())


@pytest.fixture
def client():
    with TestClient(create_app(ROOT / "artifacts")) as instance:
        yield instance


def test_api_matches_saved_pipeline(client, payload):
    model, manifest = load_artifact(ROOT / "artifacts")
    expected = model.predict_proba(ScoreRequest(**payload).to_frame())[0, 1]
    result = client.post("/predict", json=payload)
    assert result.status_code == 200
    result = result.json()
    assert result["probability"] == pytest.approx(expected, abs=1e-12)
    assert result["risk_score"] == round(100 * expected, 2)
    assert result["model_version"] == manifest["model_version"]
    assert result["target"] == TARGET


@pytest.mark.parametrize("change", [
    {"credit_limit_ntd": 0}, {"payment_amounts_ntd": [-1] * 6},
    {"repayment_status": [0] * 5}, {"repayment_status": [0.5] * 6},
    {"repayment_status": [10] * 6}, {"SEX": 1},
    {"bill_amounts_ntd": ["NaN"] * 6}, {"dataset_context": "microfinance"},
    {"credit_limit_ntd": True}, {"credit_limit_ntd": "200000"},
    {"bill_amounts_ntd": [True] * 6}, {"bill_amounts_ntd": ["100"] * 6},
    {"payment_amounts_ntd": [False] * 6}, {"payment_amounts_ntd": ["100"] * 6},
])
def test_invalid_requests_are_rejected(client, payload, change):
    response = client.post("/predict", json={**payload, **change})
    assert response.status_code == 422


def test_missing_data_is_not_replaced_with_a_score(client, payload):
    del payload["payment_amounts_ntd"]
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_bill_balance_is_allowed_and_history_order_preserved(client, payload):
    payload["bill_amounts_ntd"][0] = -10
    payload["repayment_status"] = [-2, -1, 0, 1, 2, 3]
    frame = ScoreRequest(**payload).to_frame()
    assert frame[PAY_COLUMNS].iloc[0].tolist() == payload["repayment_status"]
    assert client.post("/predict", json=payload).status_code == 200


def test_metrics_count_errors_without_storing_inputs(client, payload):
    client.post("/predict", json=payload)
    client.post("/predict", json={})
    stats = client.get("/metrics").json()
    assert stats["prediction_requests"] == 2
    assert stats["prediction_successes"] == 1
    assert stats["prediction_errors"] == 1
    assert "credit_limit_ntd" not in stats


def test_outside_training_range_is_visible(client, payload):
    payload["credit_limit_ntd"] = 100000000
    response = client.post("/predict", json=payload).json()
    assert any("LIMIT_BAL" in warning for warning in response["input_warnings"])


def test_corrupt_artifact_refuses_to_load(tmp_path):
    (tmp_path / "manifest.json").write_bytes((ROOT / "artifacts/manifest.json").read_bytes())
    (tmp_path / "model.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(tmp_path)


def test_missing_artifact_prevents_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path)):
            pass


def test_duplicate_profiles_never_cross_splits():
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(rng.integers(1, 20, (200, len(FEATURES))), columns=FEATURES)
    frame[TARGET] = np.arange(200) % 2
    frame = pd.concat([frame, frame.iloc[:30]], ignore_index=True)
    frame["ID"] = np.arange(len(frame))
    splits = make_splits(frame)
    assert splits.groupby("profile_group").split.nunique().max() == 1
    assert set(splits.split) == {"train", "calibration", "validation", "test"}
    pd.testing.assert_frame_equal(splits, make_splits(frame))


def test_saved_selection_and_partition_audit():
    splits = pd.read_csv(ROOT / "artifacts/splits.csv", dtype={"profile_group": str})
    assert len(splits) == 30000 and splits.ID.nunique() == 30000
    assert splits.groupby("profile_group").split.nunique().max() == 1
    selection = json.loads((ROOT / "artifacts/selection.json").read_text())
    metrics = json.loads((ROOT / "artifacts/metrics.json").read_text())
    expected = min(selection["validation"], key=lambda name: (
        selection["validation"][name]["log_loss"], selection["validation"][name]["brier"], name))
    assert selection["selected_model"] == metrics["selected_model"] == expected
