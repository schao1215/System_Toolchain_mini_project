"""Forecast-residual monitor, extended from one channel to all 52.

Quoted recipe:

    "Inside each run, build a table whose features are the standardized rows
    at t-1 and t-2 (104 columns) and whose targets are the standardized row
    at t (52 columns). The first two samples of each run have no score."

    "Fit one Ridge(alpha=1.0) on the training runs, predicting all 52 targets
    at once."

    "On the training runs, compute each channel's residual standard deviation
    (ddof=1)."

    "For every scored row, divide each channel's residual by that standard
    deviation, square, and sum over the 52 channels. That sum is the score."

Feature order follows the sentence "at t-1 and t-2": columns 0:52 are the
row at t-1, columns 52:104 are the row at t-2. Lags never cross a run, and
they are only formed when the sample numbers are consecutive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from miniproject.config import RIDGE_ALPHA


def lag_design(z: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the one-step design inside each run.

    Returns X with shape (n, 104), Y with shape (n, 52), and the meta rows
    of time t. Samples 1 and 2 of a run are absent, because t-2 does not exist.
    """
    if len(z) != len(meta):
        raise ValueError("z and meta must describe the same rows.")
    order = np.lexsort(
        (
            meta["sample"].to_numpy(),
            meta["simulationRun"].to_numpy(),
            meta["faultNumber"].to_numpy(),
        )
    )
    z = z[order]
    meta = meta.iloc[order].reset_index(drop=True)
    fault = meta["faultNumber"].to_numpy()
    run = meta["simulationRun"].to_numpy()
    sample = meta["sample"].to_numpy()

    # A row at index i is time t. It is usable only when i-1 is t-1 and i-2
    # is t-2 in the same fault and the same run.
    usable = np.zeros(len(meta), dtype=bool)
    usable[2:] = (
        (fault[2:] == fault[1:-1])
        & (fault[2:] == fault[:-2])
        & (run[2:] == run[1:-1])
        & (run[2:] == run[:-2])
        & (sample[2:] == sample[1:-1] + 1)
        & (sample[2:] == sample[:-2] + 2)
    )
    idx = np.flatnonzero(usable)
    # "features are the standardized rows at t-1 and t-2 (104 columns)"
    features = np.concatenate([z[idx - 1], z[idx - 2]], axis=1)
    targets = z[idx]
    return features, targets, meta.iloc[idx].reset_index(drop=True)


def fit_ridge(
    train_z: np.ndarray, train_meta: pd.DataFrame
) -> tuple[Ridge, np.ndarray]:
    """Fit one multi-output ridge and the training residual scale.

    The residual standard deviation is computed on the training runs only,
    with ddof=1, one value per channel. Later scores divide by this scale,
    so a channel that is hard to forecast in normal operation is not
    automatically louder.
    """
    features, targets, _ = lag_design(train_z, train_meta)
    # alpha=1.0 is the only setting the recipe names. Other Ridge arguments
    # stay at the sklearn defaults, including an intercept.
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(features, targets)
    residual = targets - model.predict(features)
    resid_std = residual.std(axis=0, ddof=1)
    if np.any(resid_std <= 0):
        raise RuntimeError("A training residual standard deviation is not positive.")
    print(
        f"Ridge  train_rows={len(targets):,}  features={features.shape[1]}  "
        f"alpha={RIDGE_ALPHA}"
    )
    return model, resid_std


def score_ridge(
    model: Ridge,
    resid_std: np.ndarray,
    z: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score validation, test, and faulty rows. Also return per-channel contributions.

    The score is the sum of squared standardized residuals. The contribution
    of channel j, used later for diagnosis, is that channel's own term:

        "Ridge: the contribution of channel j is its squared standardized residual."

    Output columns: faultNumber, simulationRun, sample, score.
    """
    features, targets, scored_meta = lag_design(z, meta)
    residual = targets - model.predict(features)
    standardized = residual / resid_std
    contribution = standardized**2
    score = contribution.sum(axis=1)

    out = scored_meta.loc[:, ["faultNumber", "simulationRun", "sample"]].copy()
    out["score"] = score
    out = out.sort_values(["faultNumber", "simulationRun", "sample"], kind="mergesort")
    out = out.reset_index(drop=True)

    # Keep contributions in the same row order as `out`.
    order = np.lexsort(
        (
            scored_meta["sample"].to_numpy(),
            scored_meta["simulationRun"].to_numpy(),
            scored_meta["faultNumber"].to_numpy(),
        )
    )
    return out, _pack_contributions(out, contribution[order])


def _pack_contributions(keys: pd.DataFrame, contribution: np.ndarray) -> pd.DataFrame:
    """Attach the 52 squared residuals beside the run keys.

    The wide column is not part of scores_ridge.parquet. diagnose.py uses it
    in memory and writes only the top five channels.
    """
    packed = keys.loc[:, ["faultNumber", "simulationRun", "sample"]].copy()
    packed["_contrib"] = list(contribution)
    return packed
