"""Shared choices from the project recipe.

The two detectors are independent, but the assignment says they share this
preprocessing, so it lives in one place:

    "Channels: all 52: xmeas_1 to xmeas_41 and xmv_1 to xmv_11"
    "Training runs: fault-free runs 1 to 300"
    "Validation runs: fault-free runs 301 to 400, used only to set thresholds"
    "Test runs: fault-free runs 401 to 500, used only to measure false alarms"
    "Faulty runs: faults 1 to 20, runs 1 to 20"
    "Standardization: subtract each channel's mean and divide by its standard
     deviation (ddof=1), both computed on the training runs"
"""

from pathlib import Path

# Project root is two levels above this file: src/miniproject/config.py
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# "Samples are 3 minutes apart, so a run is 25 hours."
SAMPLE_MINUTES = 3

# "In the faulty file, each fault is introduced one hour into the run, so
# samples 1 to 20 are normal and samples 21 onward are faulty."
FAULT_START_SAMPLE = 20  # last normal sample; scores after this sample are faulty

N_CHANNELS_MEAS = 41
N_CHANNELS_MV = 11

TRAIN_RUNS = range(1, 301)
VAL_RUNS = range(301, 401)
TEST_RUNS = range(401, 501)
FAULTY_RUNS = range(1, 21)
FAULTS = range(1, 21)

# 99th percentile on held-out normal runs. NumPy's default interpolation.
QUANTILE = 0.99

# "Keep the smallest number of components k whose eigenvalues add up to
# at least 90% of the total."
PCA_VARIANCE = 0.90

# "Fit one Ridge(alpha=1.0) on the training runs."
RIDGE_ALPHA = 1.0

# Detector names as written in the recipe: (`T2`, `SPE`, `ridge`).
DETECTORS = ("T2", "SPE", "ridge")
CONTRIBUTION_DETECTORS = ("SPE", "ridge")

# "keeping the top five channels per fault and detector"
TOP_CHANNELS = 5

# A fault is treated as caught in the report when more than this share of
# post-sample-20 rows are in alarm. The alarm rule already suppresses isolated
# spikes, so 0.10 is well above the roughly 1% threshold rate.
CAUGHT_RATE = 0.10


def measurement_names() -> list[str]:
    """Return the 52 channel names in recipe order."""
    meas = [f"xmeas_{i}" for i in range(1, N_CHANNELS_MEAS + 1)]
    mv = [f"xmv_{i}" for i in range(1, N_CHANNELS_MV + 1)]
    return meas + mv


# Downs and Vogel (1993), the usual TEP measurement table. Used only so the
# report can say what a channel is. The model itself uses the raw names.
CHANNEL_DESCRIPTION: dict[str, str] = {
    "xmeas_1": "A feed (stream 1)",
    "xmeas_2": "D feed (stream 2)",
    "xmeas_3": "E feed (stream 3)",
    "xmeas_4": "A and C feed (stream 4)",
    "xmeas_5": "recycle flow (stream 8)",
    "xmeas_6": "reactor feed rate (stream 6)",
    "xmeas_7": "reactor pressure",
    "xmeas_8": "reactor level",
    "xmeas_9": "reactor temperature",
    "xmeas_10": "purge rate (stream 9)",
    "xmeas_11": "product separator temperature",
    "xmeas_12": "product separator level",
    "xmeas_13": "product separator pressure",
    "xmeas_14": "product separator underflow (stream 10)",
    "xmeas_15": "stripper level",
    "xmeas_16": "stripper pressure",
    "xmeas_17": "stripper underflow (stream 11)",
    "xmeas_18": "stripper temperature",
    "xmeas_19": "stripper steam flow",
    "xmeas_20": "compressor work",
    "xmeas_21": "reactor cooling water outlet temperature",
    "xmeas_22": "separator cooling water outlet temperature",
    "xmeas_23": "reactor feed composition A",
    "xmeas_24": "reactor feed composition B",
    "xmeas_25": "reactor feed composition C",
    "xmeas_26": "reactor feed composition D",
    "xmeas_27": "reactor feed composition E",
    "xmeas_28": "reactor feed composition F",
    "xmeas_29": "purge composition A",
    "xmeas_30": "purge composition B",
    "xmeas_31": "purge composition C",
    "xmeas_32": "purge composition D",
    "xmeas_33": "purge composition E",
    "xmeas_34": "purge composition F",
    "xmeas_35": "purge composition G",
    "xmeas_36": "purge composition H",
    "xmeas_37": "product composition D",
    "xmeas_38": "product composition E",
    "xmeas_39": "product composition F",
    "xmeas_40": "product composition G",
    "xmeas_41": "product composition H",
    "xmv_1": "D feed flow valve (stream 2)",
    "xmv_2": "E feed flow valve (stream 3)",
    "xmv_3": "A feed flow valve (stream 1)",
    "xmv_4": "A and C feed flow valve (stream 4)",
    "xmv_5": "compressor recycle valve",
    "xmv_6": "purge valve (stream 9)",
    "xmv_7": "separator liquid flow valve (stream 10)",
    "xmv_8": "stripper liquid product valve (stream 11)",
    "xmv_9": "stripper steam valve",
    "xmv_10": "reactor cooling water flow valve",
    "xmv_11": "condenser cooling water flow valve",
}

# Chiang, Russell and Braatz (2000), Table 1. IDV 1 to 20.
FAULT_DESCRIPTION: dict[int, str] = {
    1: "A/C feed ratio, B composition constant (stream 4); step",
    2: "B composition, A/C ratio constant (stream 4); step",
    3: "D feed temperature (stream 2); step",
    4: "reactor cooling water inlet temperature; step",
    5: "condenser cooling water inlet temperature; step",
    6: "A feed loss (stream 1); step",
    7: "C header pressure loss, reduced availability (stream 4); step",
    8: "A, B, C feed composition (stream 4); random variation",
    9: "D feed temperature (stream 2); random variation",
    10: "C feed temperature (stream 4); random variation",
    11: "reactor cooling water inlet temperature; random variation",
    12: "condenser cooling water inlet temperature; random variation",
    13: "reaction kinetics; slow drift",
    14: "reactor cooling water valve; sticking",
    15: "condenser cooling water valve; sticking",
    16: "unknown",
    17: "unknown",
    18: "unknown",
    19: "unknown",
    20: "unknown",
}

# What a contribution plot should point at if it finds the disturbance
# rather than a downstream symptom. The report compares the actual top
# channels with this list. Westerhuis, Gurden and Smilde (2000) warn that
# contribution plots smear, so a downstream channel can outrank the source.
FAULT_EXPECTED_CHANNELS: dict[int, set[str]] = {
    1: {"xmeas_1", "xmeas_4", "xmv_3", "xmv_4", "xmeas_23", "xmeas_25", "xmeas_29", "xmeas_31"},
    2: {"xmeas_4", "xmv_4", "xmeas_24", "xmeas_30"},
    3: {"xmeas_2", "xmv_1"},
    4: {"xmeas_9", "xmeas_21", "xmv_10"},
    5: {"xmeas_11", "xmeas_13", "xmeas_22", "xmv_11"},
    6: {"xmeas_1", "xmv_3"},
    7: {"xmeas_4", "xmv_4", "xmeas_7"},
    8: {"xmeas_4", "xmv_4", "xmeas_23", "xmeas_24", "xmeas_25", "xmeas_29", "xmeas_30", "xmeas_31"},
    9: {"xmeas_2", "xmv_1"},
    10: {"xmeas_4", "xmeas_18", "xmv_4"},
    11: {"xmeas_9", "xmeas_21", "xmv_10"},
    12: {"xmeas_11", "xmeas_22", "xmv_11"},
    13: {"xmeas_7", "xmeas_9", "xmeas_16", "xmeas_18"},
    14: {"xmeas_9", "xmeas_21", "xmv_10"},
    15: {"xmeas_11", "xmeas_22", "xmv_11"},
    16: set(),
    17: set(),
    18: set(),
    19: set(),
    20: set(),
}
