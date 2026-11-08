from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from cedar.data import DATASET, DATA_SHA256, DATA_URL, FEATURES, TARGET, load_data, make_splits, sha256
from cedar.model import candidates


def metrics(y, probability):
    return {
        "n": len(y), "event_rate": float(np.mean(y)),
        "mean_prediction": float(np.mean(probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, probability)) if np.any(y) else None,
    }


def group_intervals(y, p, groups, seed, repeats=200):
    rng = np.random.default_rng(seed)
    group_rows = pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).apply(list).tolist()
    values = {name: [] for name in ("log_loss", "brier", "roc_auc", "average_precision")}
    for _ in range(repeats):
        sampled = rng.integers(0, len(group_rows), size=len(group_rows))
        indices = np.concatenate([group_rows[i] for i in sampled])
        if len(np.unique(y[indices])) < 2:
            continue
        outcome = metrics(y[indices], p[indices])
        for name in values:
            values[name].append(outcome[name])
    return {name: np.quantile(v, [0.025, 0.975]).tolist() for name, v in values.items()}


def plots(y, predictions, winner, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    displayed = dict.fromkeys(["prior", "logistic", "hist_gradient_boosting", winner])
    palette = ["#94a3b8", "#396aa4", "#d18437", "#16775c"]
    calibration = {}
    for (name, _), color in zip(displayed.items(), palette):
        p = predictions[name]
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        calibration[name] = {"predicted": predicted.tolist(), "observed": observed.tolist()}
        axes[0].plot(predicted, observed, "o-", label=name, color=color)
        fpr, tpr, _ = roc_curve(y, p)
        axes[1].plot(fpr, tpr, color=color, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1)
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed event fraction",
                title="Probability reliability · held-out test", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="Ranking · ROC curve")
    axes[2].hist(predictions[winner][y == 0], bins=20, range=(0, 1), alpha=.65, density=True,
                 label="No recorded default", color="#396aa4")
    axes[2].hist(predictions[winner][y == 1], bins=20, range=(0, 1), alpha=.55, density=True,
                 label="Recorded default", color="#d18437")
    axes[2].set(xlabel="Selected model probability", ylabel="Density", title="Prediction distributions")
    axes[2].legend(fontsize=8)
    fig.suptitle("CEDAR BASELINE  /  UCI 350  /  Single historical cohort", fontsize=14)
    fig.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)
    return calibration


def write_card(report, manifest, output):
    selected = report["selected_model"]
    rows = []
    for name, m in report["test"].items():
        rows.append(f"| {name}{' **(선택)**' if name == selected else ''} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} |")
    selected_metrics = report["test"][selected]
    ci = report["selected_test_intervals_95"]
    text = f"""# Cedar baseline — 실제 학습·평가 결과

**선택 모델: {selected}**  
버전: `{manifest['model_version']}`  
실행: {manifest['created_at']}

## 무엇을 예측하나

UCI 350의 대만 신용카드 고객에 대해 과거 6개월 기록으로 **다음 달 미상환 여부(Y)**를 예측한다.
출력은 해당 사건의 확률 추정치이며 점수는 `확률 × 100`이다.
기존 Cedar의 농업·소액금융 대출 또는 12개월 부도 확률에 대한 검증은 아니다.
원본은 2005년의 한 코호트이며, 현대 고객·다른 국가·다른 상품에 대한 성능은 알 수 없다.

## 평가 결과

선택은 validation의 log loss로 고정한 뒤 test를 열었다. 낮은 log loss/Brier, 높은 ROC-AUC/AP가 좋다.
AP는 average precision이며 PR 곡선의 사다리꼴 면적과 다르다.

| 모델 | Log loss ↓ | Brier ↓ | ROC-AUC ↑ | AP ↑ |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

최종 평가 {selected_metrics['n']:,}건, 미상환 비율 {selected_metrics['event_rate']:.2%}.
선택 모델의 ROC-AUC 95% 구간: {ci['roc_auc'][0]:.3f}–{ci['roc_auc'][1]:.3f};
log loss 95% 구간: {ci['log_loss'][0]:.3f}–{ci['log_loss'][1]:.3f}.
금융 프로필 그룹 단위 bootstrap 200회의 탐색적 구간으로, 미래·다른 시장의 불확실성을 포함하지 않는다.

![최종 평가 진단](evaluation.png)

## 실험을 재현하는 장치

- 학습/보정/선택/평가 분할을 `artifacts/splits.csv`에 보관.
- 금융 입력이 동일한 행을 같은 구간에 배정. 구간 간 금융 프로필 중복은 0.
- 전처리는 학습 구간에만 적합. sigmoid 보정은 별도 구간에서 적합.
- 후보와 선택 규칙은 `EXPERIMENT_PLAN.md`에 명시. 최종 평가 이후 재학습하지 않음.
- 데이터 원본 SHA-256, 코드 해시, 의존성 버전, 모델 해시를 manifest에 기록.
- 모델 API가 동일한 전처리·모델을 로드. 임의의 확률이나 기본 점수로 대체하지 않음.

## 이번 결과가 증명하지 않는 것

- 원본 Y의 상세 연체 일수 정의는 충분하지 않다. 30일/90일 연체라고 해석하지 않는다.
- 승인된 기존 카드 고객 데이터이므로 신규 신청자·거절 고객을 대표한다고 가정하지 않는다.
- 미래 시간 분할, 지역 분할, 소액금융기관 외부 검증을 하지 못했다.
- 개인별 위치가 없어 기후 정보를 임의로 결합하지 않았다. 기후 정보의 추가 효과는 미검증이다.
- 민감 속성을 입력에서 제외했어도 공정성이 검증된 것은 아니다.
- 특정 조치를 하면 위험이 줄어든다는 인과 효과나 자동 승인/거절 정책은 추정하지 않는다.
- API의 관측 범위 표시는 단순한 학습 범위 검사이며 통계적 신뢰구간이 아니다.

## 다음 실험

1. 실행 시점·지역·상환 결과가 함께 있는 대출 데이터 확보와 이용 조건 검토.
2. 실제 대출 상품에 맞는 사건/기간 정의, 당시 알 수 있었던 입력으로 시간 분할.
3. 동일 모델·동일 평가 구간에서 금융 정보만 사용한 버전과 기후 정보 추가 버전 비교.
4. 운영 중 입력 변화·오류·지연 확인, 정답이 도착하면 코호트별 calibration과 성능 재평가.

## 출처

Yeh, I. (2009). Default of Credit Card Clients. UCI Machine Learning Repository. CC BY 4.0.
https://doi.org/10.24432/C55S3H  
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
"""
    (output / "MODEL_CARD.md").write_text(text, encoding="utf-8")


def train(data_path: Path, output: Path, seed=42):
    artifact_dir = output / "artifacts"
    if (artifact_dir / "manifest.json").exists():
        raise ValueError("An experiment already exists here. Use a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(exist_ok=True)
    frame = load_data(data_path)
    splits = make_splits(frame, seed)
    subsets = {name: frame.loc[splits.split == name] for name in ("train", "calibration", "validation", "test")}
    source_root = Path(__file__).resolve().parents[1]
    plan_path = source_root / "EXPERIMENT_PLAN.md"
    specs = {
        "seed": seed, "selection": "validation log loss, then Brier, then name; no refit",
        "plan_sha256": sha256(plan_path),
        "splits": {k: {"rows": len(v), "event_rate": float(v[TARGET].mean())} for k, v in subsets.items()},
    }
    splits.to_csv(artifact_dir / "splits.csv", index=False)
    train_set, calibration_set = subsets["train"], subsets["calibration"]
    models = {}
    with threadpool_limits(limits=2):
        for name, model in candidates(seed).items():
            print(f"Training {name} on {len(train_set)} records", flush=True)
            model.fit(train_set[FEATURES], train_set[TARGET])
            models[name] = model
            if name != "prior":
                calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
                calibrated.fit(calibration_set[FEATURES], calibration_set[TARGET])
                models[f"{name}_sigmoid"] = calibrated

        validation = subsets["validation"]
        validation_metrics = {name: metrics(validation[TARGET], model.predict_proba(validation[FEATURES])[:, 1])
                              for name, model in models.items()}
        winner = min(models, key=lambda name: (validation_metrics[name]["log_loss"], validation_metrics[name]["brier"], name))
        # Persist the selection before looking at test outcomes.
        (artifact_dir / "selection.json").write_text(json.dumps({**specs, "selected_model": winner, "validation": validation_metrics}, indent=2))
        print(f"Locked selection before test: {winner}", flush=True)
        test = subsets["test"]
        y = test[TARGET].to_numpy()
        predictions = {name: model.predict_proba(test[FEATURES])[:, 1] for name, model in models.items()}
        report = {
            **specs, "selected_model": winner, "validation": validation_metrics,
            "test": {name: metrics(y, p) for name, p in predictions.items()},
            "selected_test_intervals_95": group_intervals(y, predictions[winner], splits.loc[test.index, "profile_group"], seed),
            "slices": {},
        }
        for name, mask in {"recent_delay": test.PAY_0.to_numpy() >= 1,
                           "no_positive_recent_delay_code": test.PAY_0.to_numpy() <= 0}.items():
            report["slices"][name] = metrics(y[mask], predictions[winner][mask])
        report["calibration_curves"] = plots(y, predictions, winner, output)
        model_path = artifact_dir / "model.joblib"
        joblib.dump(models[winner], model_path, compress=3)
        loaded = joblib.load(model_path)
        np.testing.assert_allclose(loaded.predict_proba(test[FEATURES])[:, 1], predictions[winner], rtol=0, atol=1e-12)

    artifact_hash = sha256(model_path)
    manifest = {
        "model_version": f"cedar-v0-{artifact_hash[:12]}", "selected_model": winner,
        "created_at": datetime.now(timezone.utc).isoformat(), "model_sha256": artifact_hash,
        "dataset": DATASET, "data_url": DATA_URL, "data_sha256": DATA_SHA256,
        "dataset_license": "CC BY 4.0", "dataset_citation": "Yeh, I. (2009). https://doi.org/10.24432/C55S3H",
        "target": TARGET, "prediction_horizon": "next month; original UCI label",
        "scope": "Learning benchmark: Taiwan credit card customers, 2005; not validated for microfinance",
        "features": FEATURES, "training_ranges": {c: [float(train_set[c].min()), float(train_set[c].max())] for c in FEATURES},
        "versions": {name: importlib.metadata.version(name) for name in ["scikit-learn", "pandas", "numpy", "scipy", "joblib"]},
        "python": platform.python_version(), "plan_sha256": specs["plan_sha256"],
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "test_metrics": report["test"][winner],
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    write_card(report, manifest, output)
    print(json.dumps({"selected": winner, "test": report["test"][winner], "model_version": manifest["model_version"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    train(args.data, args.output)
