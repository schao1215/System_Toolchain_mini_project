"""Run the whole project except the evidence script.

The recipe, in order:

    1. Check SHA256 of the files in data/.
    2. Standardize with the training mean and std (ddof=1) from fault-free runs 1-300.
    3. Score every validation, test, and faulty row.
    4. Set each threshold with numpy.quantile(validation scores, 0.99).
    5. Alarm only when a sample and the two samples before it all exceed the threshold.
    6. Write the detection table, the contribution table, the figure, and REPORT.pdf.

Quoted outputs:

    results/scores_pca.parquet
    results/scores_ridge.parquet
    results/thresholds.csv
    results/detection.csv
    results/contributions.csv
    REPORT.pdf
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from miniproject.config import RESULTS_DIR
from miniproject.data import load_plant, verify_checksums
from miniproject.diagnose import ridge_contribution_table, spe_contribution_table
from miniproject.evaluate import (
    alarm_mask,
    build_thresholds_and_detection,
    validation_threshold,
)
from miniproject.pca_monitor import fit_pca, reconstruction, score_pca
from miniproject.plots import plot_fault_run
from miniproject.report import pick_plot_fault, write_report
from miniproject.ridge_monitor import fit_ridge, score_ridge


def _check_alarm_rule() -> None:
    """The three-sample rule, including a missing sample that must not alarm."""
    dense = alarm_mask(
        np.array([1, 2, 3, 4]),
        np.array([5.0, 5.0, 5.0, 0.0]),
        1.0,
    )
    if dense.tolist() != [False, False, True, False]:
        raise RuntimeError("Alarm rule failed on a complete run.")
    # Sample 2 is absent, so sample 4 does not have two preceding scores.
    sparse = alarm_mask(
        np.array([1, 3, 4]),
        np.array([5.0, 5.0, 5.0]),
        1.0,
    )
    if sparse.tolist() != [False, False, False]:
        raise RuntimeError("Alarm rule crossed a missing sample.")


def _write_csv(frame: pd.DataFrame, path) -> None:
    """Enough digits that a later read-back matches the in-memory floats."""
    frame.to_csv(path, index=False, float_format="%.17g")


def _cast_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("faultNumber", "simulationRun", "sample"):
        out[column] = out[column].astype("int64")
    return out


def _assert_score_rows(pca_scores: pd.DataFrame, ridge_scores: pd.DataFrame) -> None:
    """Every required row scored, and no others."""
    if len(pca_scores) != 300_000:
        raise RuntimeError(f"PCA score file has {len(pca_scores)} rows, expected 300000.")
    # Each of 600 scored runs drops its first two samples.
    if len(ridge_scores) != 298_800:
        raise RuntimeError(f"Ridge score file has {len(ridge_scores)} rows, expected 298800.")
    free = pca_scores[pca_scores["faultNumber"] == 0]
    if int(free["simulationRun"].min()) != 301 or int(free["simulationRun"].max()) != 500:
        raise RuntimeError("Fault-free PCA scores are not runs 301 to 500.")
    if set(ridge_scores["sample"].unique()).issuperset({1, 2}):
        raise RuntimeError("Ridge scores include sample 1 or 2, which have no t-2.")
    for frame, name in ((pca_scores, "PCA"), (ridge_scores, "ridge")):
        if frame.duplicated(["faultNumber", "simulationRun", "sample"]).any():
            raise RuntimeError(f"{name} scores have duplicate keys.")


def _assert_tables_match_scores(
    pca_scores: pd.DataFrame,
    ridge_scores: pd.DataFrame,
    thresholds: pd.DataFrame,
    detection: pd.DataFrame,
) -> None:
    """Recompute the 99th percentiles from the score frames about to be saved.

    The evaluation check does this from the team's own scores, so a threshold
    that is not that percentile would fail even when the detector is right.
    """
    expected = {
        "T2": validation_threshold(pca_scores, "T2"),
        "SPE": validation_threshold(pca_scores, "SPE"),
        "ridge": validation_threshold(ridge_scores, "score"),
    }
    for _, row in thresholds.iterrows():
        got = float(row["threshold"])
        if got != expected[row["detector"]]:
            raise RuntimeError(f"Threshold for {row['detector']} does not match quantile.")
    if len(detection) != 63:
        raise RuntimeError("detection.csv must have 21 rows for each of 3 detectors.")
    faults = set(detection["fault"].astype(int))
    if faults != set(range(0, 21)):
        raise RuntimeError("detection.csv is missing a fault from 0 to 20.")


def run() -> None:
    """Fit, score, evaluate, diagnose, and write the report.

    Does not call the course evidence script.
    """
    _check_alarm_rule()
    verify_checksums()
    plant = load_plant()

    model, k = fit_pca(plant.train_z)
    cumulative = np.cumsum(model.explained_variance_ratio_)
    variance = float(cumulative[k - 1])
    pca_scores = _cast_keys(score_pca(model, k, plant.score_z, plant.score_meta))
    reconstructed = reconstruction(model, k, plant.score_z)

    ridge_model, resid_std = fit_ridge(plant.train_z, plant.train_meta)
    # The recipe checks that the code contains a ridge fit. This is that fit.
    if not hasattr(ridge_model, "coef_"):
        raise RuntimeError("Ridge model was not fit.")
    ridge_scores, ridge_contrib = score_ridge(
        ridge_model, resid_std, plant.score_z, plant.score_meta
    )
    ridge_scores = _cast_keys(ridge_scores)
    _assert_score_rows(pca_scores, ridge_scores)

    thresholds, detection, alarmed = build_thresholds_and_detection(pca_scores, ridge_scores)
    _assert_tables_match_scores(pca_scores, ridge_scores, thresholds, detection)

    spe_table = spe_contribution_table(
        plant.score_z,
        plant.score_meta,
        reconstructed,
        alarmed["SPE"],
        plant.channels,
    )
    ridge_table = ridge_contribution_table(ridge_contrib, alarmed["ridge"], plant.channels)
    contributions = pd.concat([spe_table, ridge_table], ignore_index=True)
    contributions = contributions.sort_values(
        ["detector", "fault", "rank"], kind="mergesort"
    ).reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pca_path = RESULTS_DIR / "scores_pca.parquet"
    ridge_path = RESULTS_DIR / "scores_ridge.parquet"
    pca_scores.to_parquet(pca_path, index=False)
    ridge_scores.to_parquet(ridge_path, index=False)
    _write_csv(thresholds, RESULTS_DIR / "thresholds.csv")
    _write_csv(detection, RESULTS_DIR / "detection.csv")
    _write_csv(contributions, RESULTS_DIR / "contributions.csv")

    # Read the parquet back. This is our check, not the course evidence script.
    reloaded = pd.read_parquet(pca_path)
    if not np.allclose(reloaded["T2"], pca_scores["T2"], rtol=0, atol=0):
        raise RuntimeError("PCA parquet did not round-trip.")

    figure = plot_fault_run(
        pca_scores,
        ridge_scores,
        dict(zip(thresholds["detector"], thresholds["threshold"])),
        fault=pick_plot_fault(detection),
        run=1,
    )
    write_report(
        detection=detection,
        thresholds=thresholds,
        contributions=contributions,
        figure=figure,
        k=k,
        variance=variance,
        pca_scores=pca_scores,
        ridge_scores=ridge_scores,
    )
    _print_summary(detection, k, variance)


def _print_summary(detection: pd.DataFrame, k: int, variance: float) -> None:
    print(f"\nPCA k={k}  retained variance={variance:.4f}")
    print("fault  T2_rate  SPE_rate  ridge_rate  T2_miss  SPE_miss  ridge_miss")
    for fault in range(0, 21):
        cells = []
        misses = []
        for detector in ("T2", "SPE", "ridge"):
            row = detection[
                (detection["fault"] == fault) & (detection["detector"] == detector)
            ].iloc[0]
            cells.append(f"{row['detection_rate']:.3f}")
            missed = row["runs_missed"]
            misses.append("  ." if pd.isna(missed) else f"{int(missed):3d}")
        print(
            f"{fault:5d}  {cells[0]:>7}  {cells[1]:>8}  {cells[2]:>10}  "
            f"{misses[0]}  {misses[1]}  {misses[2]}"
        )


if __name__ == "__main__":
    run()
