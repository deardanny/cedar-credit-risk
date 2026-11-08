from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from cedar.data import BILL_COLUMNS, DATASET, FEATURES, PAID_COLUMNS, PAY_COLUMNS, TARGET, validate_features
from cedar.model import load_artifact

Amount = Annotated[FiniteFloat, Field(ge=0)]
Status = Annotated[int, Field(strict=True, ge=-2, le=9)]


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_context: Literal["uci_taiwan_credit_card_2005"]
    credit_limit_ntd: Annotated[FiniteFloat, Field(gt=0)]
    repayment_status: Annotated[list[Status], Field(min_length=6, max_length=6)]
    bill_amounts_ntd: Annotated[list[FiniteFloat], Field(min_length=6, max_length=6)]
    payment_amounts_ntd: Annotated[list[Amount], Field(min_length=6, max_length=6)]

    def to_frame(self) -> pd.DataFrame:
        row = {"LIMIT_BAL": self.credit_limit_ntd}
        row.update(zip(PAY_COLUMNS, self.repayment_status))
        row.update(zip(BILL_COLUMNS, self.bill_amounts_ntd))
        row.update(zip(PAID_COLUMNS, self.payment_amounts_ntd))
        frame = pd.DataFrame([row], columns=FEATURES)
        validate_features(frame)
        return frame


class ScoreResponse(BaseModel):
    model_version: str
    target: str
    probability: float
    risk_score: float
    score_definition: str
    scope: str
    input_warnings: list[str]


def create_app(artifact_directory: Path | None = None) -> FastAPI:
    directory = artifact_directory or Path(os.environ.get(
        "CEDAR_ARTIFACT_DIR", Path(__file__).resolve().parents[1] / "artifacts"))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # No fallback score: missing/incompatible artifacts prevent service readiness.
        application.state.model, application.state.manifest = load_artifact(directory)
        yield

    application = FastAPI(
        title="Cedar baseline scoring API", version="0.1.0", lifespan=lifespan,
        description="UCI 350 learning benchmark. Arrays run from September back to April 2005. "
                    "Amounts are in New Taiwan dollars. This endpoint is not validated for MFI lending.",
    )
    lock = threading.Lock()
    counters = {"prediction_requests": 0, "prediction_errors": 0, "prediction_successes": 0,
                "total_latency_ms": 0.0, "probability_sum": 0.0}

    @application.middleware("http")
    async def observe(request: Request, call_next):
        if request.url.path != "/predict":
            return await call_next(request)
        start = perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            with lock:
                counters["prediction_requests"] += 1
                counters["prediction_errors"] += int(status >= 400)
                counters["total_latency_ms"] += (perf_counter() - start) * 1000

    @application.get("/health")
    def health():
        return {"status": "ready", "model_version": application.state.manifest["model_version"]}

    @application.get("/model")
    def model_info():
        manifest = application.state.manifest
        return {key: manifest[key] for key in ["model_version", "selected_model", "target", "scope", "test_metrics", "features"]}

    @application.get("/metrics")
    def service_metrics():
        with lock:
            result = dict(counters)
        result["mean_latency_ms"] = result["total_latency_ms"] / max(result["prediction_requests"], 1)
        result["mean_probability"] = result["probability_sum"] / max(result["prediction_successes"], 1)
        result["note"] = "In-process aggregate metrics; reset on restart. No repayment labels or performance drift measurement."
        return result

    @application.post("/predict", response_model=ScoreResponse)
    def predict(payload: ScoreRequest):
        frame = payload.to_frame()
        manifest = application.state.manifest
        warnings = []
        for column, (minimum, maximum) in manifest["training_ranges"].items():
            if not minimum <= frame.iloc[0][column] <= maximum:
                warnings.append(f"{column} is outside the observed training range.")
        probability = float(application.state.model.predict_proba(frame)[0, 1])
        with lock:
            counters["prediction_successes"] += 1
            counters["probability_sum"] += probability
        return ScoreResponse(
            model_version=manifest["model_version"], target=TARGET,
            probability=probability, risk_score=round(100 * probability, 2),
            score_definition="100 × P(original UCI default-payment-next-month label = 1)",
            scope=manifest["scope"], input_warnings=warnings,
        )

    return application


app = create_app()
