"""Load the two TEP parquet files and standardize on the training runs.

Quoted recipe:

    "Rows to score: every row of the validation, test and faulty runs."
    "Fault-free rows have faultNumber 0."
    "The evidence script refuses files that do not match."

Training rows are used to fit the models and the scaler. They are not written
into the score files.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from miniproject.config import (
    DATA_DIR,
    FAULT_START_SAMPLE,
    measurement_names,
)


@dataclass
class PlantData:
    """Standardized blocks plus the run labels needed to score and split them."""

    channels: list[str]
    mean: np.ndarray
    std: np.ndarray
    # Each block is already standardized with the training mean and std.
    train_z: np.ndarray
    train_meta: pd.DataFrame
    score_z: np.ndarray
    score_meta: pd.DataFrame


def verify_checksums(data_dir=DATA_DIR) -> None:
    """Refuse the local files if they do not match the published SHA256SUMS.

    The assignment says the evidence script refuses files that do not match.
    This check runs first so a bad download cannot silently train a model.
    """
    sums_path = data_dir / "SHA256SUMS"
    if not sums_path.exists():
        raise FileNotFoundError(
            f"Missing {sums_path}. Download the files in the assignment into data/."
        )
    for line in sums_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split()
        name = name.lstrip("*")
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        got = hasher.hexdigest()
        if got != digest:
            raise ValueError(f"Checksum mismatch for {name}: {got} != {digest}")
        print(f"checksum ok  {name}")


def _read_parquet(path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    # Column names in the hosted files are expected to be xmeas_1, xmv_1,
    # simulationRun, sample, and (on the faulty file) faultNumber.
    frame.columns = [str(column) for column in frame.columns]
    return frame


def _require_channels(frame: pd.DataFrame, channels: list[str]) -> None:
    missing = [name for name in channels if name not in frame.columns]
    if missing:
        raise KeyError(
            "Parquet file is missing channels the recipe asks for: "
            + ", ".join(missing[:8])
        )


def _meta(frame: pd.DataFrame, fault_number: np.ndarray | None = None) -> pd.DataFrame:
    """Keep the three identifier columns the score files must contain."""
    if fault_number is None:
        if "faultNumber" not in frame.columns:
            fault_number = np.zeros(len(frame), dtype=np.int64)
        else:
            fault_number = frame["faultNumber"].to_numpy(dtype=np.int64)
    return pd.DataFrame(
        {
            "faultNumber": np.asarray(fault_number, dtype=np.int64),
            "simulationRun": frame["simulationRun"].to_numpy(dtype=np.int64),
            "sample": frame["sample"].to_numpy(dtype=np.int64),
        }
    )


def load_plant(data_dir=DATA_DIR) -> PlantData:
    """Read both files, split the runs, and standardize with training ddof=1.

    Quoted rule: "subtract each channel's mean and divide by its standard
    deviation (ddof=1), both computed on the training runs."

    NumPy's default std uses ddof=0, which would not match the recipe, so the
    divisor is set explicitly.
    """
    channels = measurement_names()
    free = _read_parquet(data_dir / "tep_fault_free_training.parquet")
    faulty = _read_parquet(data_dir / "tep_faulty_training_runs01-20.parquet")
    _require_channels(free, channels)
    _require_channels(faulty, channels)

    free_meta = _meta(free)
    faulty_meta = _meta(faulty)
    if not set(faulty_meta["faultNumber"].unique()).issuperset(set(range(1, 21))):
        raise ValueError("Faulty file does not contain faultNumber 1 to 20.")

    # "samples 1 to 20 are normal and samples 21 onward are faulty."
    if int(free_meta["sample"].min()) != 1 or int(free_meta["sample"].max()) != 500:
        raise ValueError("Expected fault-free sample numbers 1 to 500.")
    if FAULT_START_SAMPLE != 20:
        raise ValueError("Fault injection sample drifted from the recipe.")

    free_x = free[channels].to_numpy(dtype=np.float64)
    faulty_x = faulty[channels].to_numpy(dtype=np.float64)

    train_mask = free_meta["simulationRun"].between(1, 300).to_numpy()
    val_mask = free_meta["simulationRun"].between(301, 400).to_numpy()
    test_mask = free_meta["simulationRun"].between(401, 500).to_numpy()
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Fault-free file is missing training, validation, or test runs.")

    # Mean and std come from training runs only. Validation is reserved for
    # thresholds, and the test runs are reserved for the false-alarm rate.
    train_x = free_x[train_mask]
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0, ddof=1)
    if np.any(std == 0):
        dead = [channels[i] for i in np.flatnonzero(std == 0)]
        raise ValueError(f"Zero training std for {dead}")

    def standardize(values: np.ndarray) -> np.ndarray:
        return (values - mean) / std

    train_z = standardize(train_x)
    score_z = np.vstack(
        [
            standardize(free_x[val_mask]),
            standardize(free_x[test_mask]),
            standardize(faulty_x),
        ]
    )
    score_meta = pd.concat(
        [
            free_meta.loc[val_mask],
            free_meta.loc[test_mask],
            faulty_meta,
        ],
        ignore_index=True,
    )
    # Fault-free scored rows are validation and test. Force faultNumber 0
    # even if the source file stored something else.
    score_meta.loc[score_meta.index[: val_mask.sum() + test_mask.sum()], "faultNumber"] = 0
    score_meta["faultNumber"] = score_meta["faultNumber"].astype(np.int64)

    train_meta = free_meta.loc[train_mask].reset_index(drop=True)
    train_meta["faultNumber"] = np.int64(0)

    print(
        "rows  "
        f"train={len(train_meta):,}  "
        f"to_score={len(score_meta):,}  "
        f"channels={len(channels)}"
    )
    return PlantData(
        channels=channels,
        mean=mean,
        std=std,
        train_z=train_z,
        train_meta=train_meta,
        score_z=score_z,
        score_meta=score_meta,
    )
