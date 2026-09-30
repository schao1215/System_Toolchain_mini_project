"""Per-channel contributions for SPE and the ridge score.

Quoted rule:

    "For the SPE and ridge detectors, a statistic that is a sum of squares
    splits naturally into one term per channel. That split is called a
    contribution."

    "SPE: the contribution of channel j is (z_j - (P P^T z)_j)^2."
    "Ridge: the contribution of channel j is its squared standardized residual."

    "For each fault and each of the two detectors, average the contributions
    over every sample that is in alarm after sample 20, across the 20 runs.
    ... keeping the top five channels per fault and detector."

A contribution plot points at where the fault shows up, which is not always
where it started.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from miniproject.config import FAULTS, TOP_CHANNELS


def _top_from_average(
    fault: int,
    detector: str,
    average: np.ndarray,
    channels: list[str],
) -> list[dict[str, object]]:
    """Keep the five largest average contributions. Ties break on channel name."""
    order = sorted(
        range(len(channels)),
        key=lambda index: (-float(average[index]), channels[index]),
    )
    rows = []
    for rank, index in enumerate(order[:TOP_CHANNELS], start=1):
        rows.append(
            {
                "fault": fault,
                "detector": detector,
                "rank": rank,
                "channel": channels[index],
                "contribution": float(average[index]),
            }
        )
    return rows


def spe_contribution_table(
    z: np.ndarray,
    meta: pd.DataFrame,
    reconstructed: np.ndarray,
    alarms: pd.DataFrame,
    channels: list[str],
) -> pd.DataFrame:
    """Average SPE contributions on post-sample-20 alarming rows."""
    # One term per channel, as specified. Do not renormalize the row.
    contribution = (z - reconstructed) ** 2
    frame = meta.loc[:, ["faultNumber", "simulationRun", "sample"]].copy()
    frame["_row"] = np.arange(len(frame))
    alarm_keys = alarms.loc[alarms["alarm"], ["faultNumber", "simulationRun", "sample"]]
    merged = frame.merge(
        alarm_keys,
        on=["faultNumber", "simulationRun", "sample"],
        how="inner",
    )
    merged = merged[merged["sample"] > 20]

    rows: list[dict[str, object]] = []
    for fault in FAULTS:
        part = merged[merged["faultNumber"] == fault]
        if part.empty:
            print(f"contributions  SPE fault {fault}: no alarming samples after sample 20")
            continue
        average = contribution[part["_row"].to_numpy()].mean(axis=0)
        rows.extend(_top_from_average(fault, "SPE", average, channels))
    return pd.DataFrame(
        rows, columns=["fault", "detector", "rank", "channel", "contribution"]
    )


def ridge_contribution_table(
    packed: pd.DataFrame,
    alarms: pd.DataFrame,
    channels: list[str],
) -> pd.DataFrame:
    """Average ridge contributions on post-sample-20 alarming rows.

    `packed` carries one length-52 vector per scored row in `_contrib`.
    """
    alarm_keys = alarms.loc[alarms["alarm"], ["faultNumber", "simulationRun", "sample"]]
    merged = packed.merge(
        alarm_keys,
        on=["faultNumber", "simulationRun", "sample"],
        how="inner",
    )
    merged = merged[merged["sample"] > 20]

    rows: list[dict[str, object]] = []
    for fault in FAULTS:
        part = merged[merged["faultNumber"] == fault]
        if part.empty:
            print(f"contributions  ridge fault {fault}: no alarming samples after sample 20")
            continue
        stacked = np.vstack(part["_contrib"].to_list())
        average = stacked.mean(axis=0)
        rows.extend(_top_from_average(fault, "ridge", average, channels))
    return pd.DataFrame(
        rows, columns=["fault", "detector", "rank", "channel", "contribution"]
    )
