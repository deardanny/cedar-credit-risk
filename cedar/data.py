from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_URL = "https://archive.ics.uci.edu/static/public/350/data.csv"
DATA_SHA256 = "45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0"
DATASET = "uci_taiwan_credit_card_2005"
TARGET = "default_payment_next_month"
PAY_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAID_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
AMOUNT_COLUMNS = ["LIMIT_BAL", *BILL_COLUMNS, *PAID_COLUMNS]
FEATURES = [*AMOUNT_COLUMNS, *PAY_COLUMNS]
COLUMN_NAMES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    *PAY_COLUMNS, *BILL_COLUMNS, *PAID_COLUMNS,
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(path: Path) -> None:
    if path.exists():
        if sha256(path) != DATA_SHA256:
            raise ValueError("Existing data checksum differs from the approved snapshot.")
        return
    with urlopen(DATA_URL, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
        raise ValueError("UCI data changed; review the new version before using it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def load_data(path: Path) -> pd.DataFrame:
    if sha256(path) != DATA_SHA256:
        raise ValueError("Dataset fingerprint mismatch.")
    raw = pd.read_csv(path)
    expected = {"ID", "Y", *[f"X{i}" for i in range(1, 24)]}
    if set(raw.columns) != expected or len(raw) != 30000:
        raise ValueError("Unexpected UCI dataset schema or size.")
    if raw.isna().any().any() or raw.ID.duplicated().any():
        raise ValueError("Unexpected missing values or duplicate record IDs.")
    if set(raw.Y.unique()) != {0, 1}:
        raise ValueError("The target must contain both binary classes.")
    rename = {f"X{i}": name for i, name in enumerate(COLUMN_NAMES, 1)}
    frame = raw.rename(columns={**rename, "Y": TARGET})
    validate_features(frame[FEATURES])
    return frame


def validate_features(frame: pd.DataFrame) -> None:
    if list(frame.columns) != FEATURES:
        raise ValueError("Feature schema/order mismatch.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite and complete.")
    if (frame.LIMIT_BAL <= 0).any() or (frame[PAID_COLUMNS] < 0).any().any():
        raise ValueError("Credit limits must be positive and payments non-negative.")
    pay = frame[PAY_COLUMNS].to_numpy(dtype=float)
    if not ((pay >= -2) & (pay <= 9) & (pay == np.floor(pay))).all():
        raise ValueError("Repayment status must be an integer category from -2 to 9.")


def make_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    # Group identical financial profiles to avoid memorizing duplicates across splits.
    # All rows refer to one historical cohort; this is NOT a temporal backtest.
    groups = pd.util.hash_pandas_object(frame[FEATURES], index=False).astype(str)
    folds = np.full(len(frame), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(splitter.split(frame[FEATURES], frame[TARGET], groups)):
        folds[indices] = fold
    split = np.select(
        [folds < 6, folds == 6, folds == 7],
        ["train", "calibration", "validation"], default="test",
    )
    result = pd.DataFrame({"ID": frame.ID, "profile_group": groups, "fold": folds, "split": split})
    if result.groupby("profile_group").split.nunique().max() != 1:
        raise AssertionError("A profile group leaked across splits.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/uci350.csv"))
    args = parser.parse_args()
    download(args.output)
    print(f"Verified UCI snapshot: {args.output} ({DATA_SHA256})")
