"""PCA monitor: T2 inside the normal pattern, SPE outside it.

Quoted recipe:

    1. Standardize the training data. Compute the principal components of its
       covariance matrix (on standardized data this is the correlation matrix).
    2. Keep the smallest number of components k whose eigenvalues add up to
       at least 90% of the total.
    3. For each standardized row z, with P the k retained loading vectors and
       lambda_i their eigenvalues:

           t = P^T z
           T^2 = sum_{i=1}^{k} t_i^2 / lambda_i
           SPE = || z - P P^T z ||^2

The assignment also says sklearn.decomposition.PCA gives the same model:
fit on the standardized training data, use explained_variance_ for lambda_i,
and inverse_transform(transform(z)) for P P^T z. This file follows that model
with svd_solver="full", so the components are the exact eigenvectors rather
than a randomized approximation.

T2 asks: "Is it unusual within the normal pattern?"
SPE asks: "Is it unusual outside the normal pattern?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from miniproject.config import PCA_VARIANCE


def fit_pca(train_z: np.ndarray) -> tuple[PCA, int]:
    """Fit PCA on standardized training rows and choose k at 90% variance.

    train_z must already be standardized with the training mean and std.
    Its column mean is then zero, so a second centering inside PCA does not
    move the loadings. k is the smallest count with cumulative eigenvalue
    mass at least 0.90. searchsorted(..., side="left") implements "at least":
    an eigenvalue that lands exactly on 90% is kept.
    """
    if train_z.ndim != 2:
        raise ValueError("train_z must be a 2-d array of standardized rows.")
    model = PCA(svd_solver="full").fit(train_z)
    cumulative = np.cumsum(model.explained_variance_ratio_)
    k = int(np.searchsorted(cumulative, PCA_VARIANCE, side="left") + 1)
    if k < 1 or k > train_z.shape[1]:
        raise RuntimeError(f"PCA retained an impossible k={k}.")
    # Guard the "smallest k" claim: k-1 components, if any, are not enough.
    if k > 1 and cumulative[k - 2] >= PCA_VARIANCE:
        raise RuntimeError("k is not the smallest 90% cutoff.")
    print(
        f"PCA  k={k}  variance={cumulative[k - 1]:.4f}  "
        f"of {train_z.shape[1]} channels"
    )
    return model, k


def score_pca(model: PCA, k: int, z: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """Score every supplied row. Caller must pass only validation, test, and faulty rows.

    Quoted output: results/scores_pca.parquet with columns
    faultNumber, simulationRun, sample, T2, SPE.
    """
    # components_ rows are the loading vectors. P in the recipe is those
    # vectors as columns, so P^T z is z @ components_[:k].T.
    components = model.components_[:k]
    lam = model.explained_variance_[:k]
    if np.any(lam <= 0):
        raise RuntimeError("A retained PCA eigenvalue is not positive.")

    # Training rows were centered before PCA, and PCA subtracts that mean
    # again inside transform. The recipe's z is the standardized row, whose
    # training mean is zero. Project z directly so SPE matches
    # ||z - P P^T z||^2 and not a second recentering.
    scores = z @ components.T
    reconstructed = scores @ components
    t2 = np.sum(scores**2 / lam, axis=1)
    spe = np.sum((z - reconstructed) ** 2, axis=1)

    out = meta.loc[:, ["faultNumber", "simulationRun", "sample"]].copy()
    out["T2"] = t2
    out["SPE"] = spe
    out = out.sort_values(["faultNumber", "simulationRun", "sample"], kind="mergesort")
    out = out.reset_index(drop=True)
    return out


def reconstruction(model: PCA, k: int, z: np.ndarray) -> np.ndarray:
    """Return P P^T z for contribution plots.

    SPE contribution of channel j is (z_j - (P P^T z)_j)^2.
    """
    components = model.components_[:k]
    scores = z @ components.T
    return scores @ components
