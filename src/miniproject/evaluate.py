"""Thresholds, three-sample alarms, and the detection table.

Quoted rules:

    "For each of the three statistics (T2, SPE, ridge), the threshold is
    numpy.quantile(scores, 0.99) over the validation runs, with NumPy's
    default interpolation."

    "A sample is in alarm when it and the two samples before it all exceed
    the threshold. Alarms never cross a run boundary."

    "false-alarm rate: the share of samples in alarm over the test runs,
    reported as fault 0"

    "detection rate: for each faulty run, the share of samples after sample
    20 that are in alarm, averaged over the 20 runs"

    "detection delay: for each faulty run, minutes from sample 20 to the
    first alarm after it; report the median over the runs that alarmed,
    and how many runs never alarmed."

The evaluation checks recompute this table from the team's own scores, so
the formulas here are the ones written back to results/detection.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from miniproject.config import FAULTS, QUANTILE, SAMPLE_MINUTES


def alarm_mask(samples: np.ndarray, values: np.ndarray, threshold: float) -> np.ndarray:
    """Mark samples whose own score and the two previous samples all exceed the threshold.

    "Exceed" is strict: a score equal to the quantile is not an alarm.
    If sample s-1 or s-2 is missing, as with the first two ridge samples,
    that row cannot alarm. The caller must pass a single run.
    """
    above = values > threshold
    by_sample = {int(sample): bool(flag) for sample, flag in zip(samples, above)}
    return np.fromiter(
        (
            by_sample.get(int(sample), False)
            and by_sample.get(int(sample) - 1, False)
            and by_sample.get(int(sample) - 2, False)
            for sample in samples
        ),
        dtype=bool,
        count=len(samples),
    )


def attach_alarms(scores: pd.DataFrame, value_col: str, threshold: float) -> pd.DataFrame:
    """Add an alarm column. The window is reset at every (faultNumber, simulationRun)."""
    parts: list[pd.DataFrame] = []
    grouped = scores.groupby(["faultNumber", "simulationRun"], sort=False)
    for _, part in grouped:
        part = part.sort_values("sample", kind="mergesort").copy()
        part["alarm"] = alarm_mask(
            part["sample"].to_numpy(),
            part[value_col].to_numpy(dtype=np.float64),
            threshold,
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def validation_threshold(scores: pd.DataFrame, value_col: str) -> float:
    """99th percentile of fault-free validation runs 301 to 400.

    np.quantile is called with no method argument so the interpolation is
    whatever NumPy's default is on this interpreter.
    """
    validation = scores[
        (scores["faultNumber"] == 0) & (scores["simulationRun"].between(301, 400))
    ]
    if validation.empty:
        raise ValueError(f"No validation rows for {value_col}.")
    return float(np.quantile(validation[value_col].to_numpy(dtype=np.float64), QUANTILE))


def detection_rows(alarmed: pd.DataFrame, detector: str) -> list[dict[str, object]]:
    """One fault-0 row plus faults 1 to 20 for a single detector.

    Fault 0 stores the false-alarm rate in detection_rate. Delay and
    runs_missed are defined only "for each faulty run", so those two cells
    are left empty on the fault-0 row.
    """
    rows: list[dict[str, object]] = []
    test = alarmed[
        (alarmed["faultNumber"] == 0) & (alarmed["simulationRun"].between(401, 500))
    ]
    if test.empty:
        raise ValueError(f"No test rows for {detector}.")
    # "the share of samples in alarm over the test runs"
    rows.append(
        {
            "fault": 0,
            "detector": detector,
            "detection_rate": float(test["alarm"].mean()),
            "median_delay_min": np.nan,
            "runs_missed": np.nan,
        }
    )

    for fault in FAULTS:
        sub = alarmed[alarmed["faultNumber"] == fault]
        runs = sorted(int(run) for run in sub["simulationRun"].unique())
        if len(runs) != 20:
            raise ValueError(f"Fault {fault} has {len(runs)} runs, expected 20.")
        per_run_rate: list[float] = []
        delays: list[float] = []
        missed = 0
        for run in runs:
            part = sub[sub["simulationRun"] == run]
            # "the share of samples after sample 20 that are in alarm"
            after = part[part["sample"] > 20]
            if after.empty:
                raise ValueError(f"Fault {fault} run {run} has no samples after 20.")
            per_run_rate.append(float(after["alarm"].mean()))
            hit = after.loc[after["alarm"], "sample"]
            if hit.empty:
                missed += 1
                continue
            # "minutes from sample 20 to the first alarm after it"
            # Sample 21 is one step (3 minutes) after sample 20.
            first = int(hit.min())
            delays.append(float((first - 20) * SAMPLE_MINUTES))
        rows.append(
            {
                "fault": fault,
                "detector": detector,
                "detection_rate": float(np.mean(per_run_rate)),
                "median_delay_min": float(np.median(delays)) if delays else np.nan,
                "runs_missed": int(missed),
            }
        )
    return rows


def build_thresholds_and_detection(
    pca_scores: pd.DataFrame,
    ridge_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Thresholds, the detection table, and alarmed frames for diagnosis."""
    specs = (
        ("T2", pca_scores, "T2"),
        ("SPE", pca_scores, "SPE"),
        ("ridge", ridge_scores, "score"),
    )
    threshold_rows = []
    detection: list[dict[str, object]] = []
    alarmed: dict[str, pd.DataFrame] = {}
    for detector, frame, column in specs:
        threshold = validation_threshold(frame, column)
        threshold_rows.append({"detector": detector, "threshold": threshold})
        marked = attach_alarms(frame, column, threshold)
        alarmed[detector] = marked
        detection.extend(detection_rows(marked, detector))
        print(f"threshold  {detector:5s}  {threshold:.6g}")

    thresholds = pd.DataFrame(threshold_rows, columns=["detector", "threshold"])
    detection_frame = pd.DataFrame(detection)[
        ["fault", "detector", "detection_rate", "median_delay_min", "runs_missed"]
    ]
    return thresholds, detection_frame, alarmed
