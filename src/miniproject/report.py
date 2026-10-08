"""Build REPORT.pdf from the tables and the fault-run figure.

The assignment fixes the section order and a four-page limit, plus an
appendix that does not count toward those four pages:

    1. The plant and the task.
    2. The two detectors.
    3. Results.
    4. Comparison.
    5. Faults nobody catches.
    6. Diagnosis.
    7. Limits.
    8. AI use.
    Appendix: contributions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fpdf import FPDF

from miniproject.config import (
    CAUGHT_RATE,
    CHANNEL_DESCRIPTION,
    FAULT_DESCRIPTION,
    ROOT,
    SAMPLE_MINUTES,
)
from miniproject.evaluate import attach_alarms

SAMPLES_PER_DAY = 24 * 60 // SAMPLE_MINUTES


# One entry per member for the contributions appendix, in role order.
TEAM = [
    {
        "name": "Xinran Li",
        "andrew": "xinranl3",
        "role": "Part 1, PCA monitor (T2 and SPE)",
        "code": "data.py (checksums, ddof = 1 standardization), pca_monitor.py, plots.py",
    },
    {
        "name": "Mingyao Xu",
        "andrew": "mingyaox",
        "role": "Part 2, forecast-residual monitor (ridge on lags t-1, t-2)",
        "code": "ridge_monitor.py (lag_design, fit_ridge, score_ridge)",
    },
    {
        "name": "Mark",
        "andrew": "marktan",
        "role": "Part 3, evaluation and submission (thresholds, alarms, detection table)",
        "code": (
            "evaluate.py (validation_threshold, alarm_mask, detection_rows), the "
            "self-checks in pipeline.py (_check_alarm_rule, _assert_score_rows); ran the "
            "course evidence script"
        ),
    },
    {
        "name": "Sean Chao",
        "andrew": "hsuanlec",
        "role": "Part 4, diagnosis (contributions and the fault list)",
        "code": "diagnose.py; FAULT_DESCRIPTION and FAULT_EXPECTED_CHANNELS in config.py",
    },
]

# Vertical space after a paragraph and before a section heading, in mm.
PARAGRAPH_GAP = 2.5
SECTION_GAP = 3.0


def _rate(detection: pd.DataFrame, fault: int, detector: str) -> float:
    row = detection[(detection["fault"] == fault) & (detection["detector"] == detector)]
    return float(row["detection_rate"].iloc[0])


def _missed(detection: pd.DataFrame, fault: int, detector: str) -> int:
    row = detection[(detection["fault"] == fault) & (detection["detector"] == detector)]
    value = row["runs_missed"].iloc[0]
    if pd.isna(value):
        return 0
    return int(value)


def _caught(detection: pd.DataFrame, detector: str) -> list[int]:
    """Faults whose detection rate is at least CAUGHT_RATE."""
    found = []
    for fault in range(1, 21):
        if _rate(detection, fault, detector) >= CAUGHT_RATE:
            found.append(fault)
    return found


def pick_plot_fault(detection: pd.DataFrame) -> int:
    """Prefer a fault that moves all three statistics, so one figure serves section 2."""
    best_fault = 1
    best_key = (-1.0, 99)
    for fault in range(1, 21):
        rates = [_rate(detection, fault, name) for name in ("T2", "SPE", "ridge")]
        key = (min(rates), -fault)
        if key > best_key:
            best_key = key
            best_fault = fault
    return best_fault


def pick_diagnosis_faults(contributions: pd.DataFrame) -> list[int]:
    """Three detected faults on different equipment, for section 6."""
    present = set(contributions.loc[contributions["detector"] == "SPE", "fault"].astype(int))
    priority = [1, 4, 6, 11, 14, 7, 2, 5, 8, 12, 13, 10, 16, 17, 18, 19, 20]
    chosen = [fault for fault in priority if fault in present]
    return chosen[:3]


def _fmt_rate(value: float) -> str:
    return f"{value:.3f}"


def _fmt_delay(detection: pd.DataFrame, fault: int, detector: str) -> str:
    row = detection[(detection["fault"] == fault) & (detection["detector"] == detector)].iloc[0]
    delay = row["median_delay_min"]
    missed = row["runs_missed"]
    if pd.isna(delay):
        return "none"
    # Medians of an even number of runs land on a half minute. Keep that half
    # instead of rounding 25.5 up to 26.
    if abs(float(delay) - round(float(delay))) < 1e-6:
        text = f"{int(round(float(delay)))} min"
    else:
        text = f"{float(delay):.1f} min"
    if not pd.isna(missed) and int(missed) > 0:
        text += f", {int(missed)} missed"
    return text


def _names(channels: list[str]) -> str:
    parts = []
    for channel in channels:
        what = CHANNEL_DESCRIPTION.get(channel, channel)
        parts.append(f"{channel} ({what})")
    return "; ".join(parts)


def _top_channels(contributions: pd.DataFrame, fault: int, detector: str) -> list[str]:
    part = contributions[
        (contributions["fault"] == fault) & (contributions["detector"] == detector)
    ].sort_values("rank")
    return part["channel"].tolist()


class Report(FPDF):
    """Letter PDF. Body sections stay within four pages; the appendix follows."""

    def __init__(self) -> None:
        super().__init__(format="Letter", unit="mm")
        self.set_auto_page_break(auto=True, margin=14)
        self.set_margins(14, 14, 14)

    def footer(self) -> None:
        self.set_y(-10)
        self.set_font("Helvetica", size=8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 4, f"{self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def h1(self, text: str) -> None:
        self.set_font("Helvetica", "B", 13)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def h2(self, text: str) -> None:
        self.set_x(self.l_margin)
        # Keep a heading with at least three lines of its section.
        if self.will_page_break(SECTION_GAP + 6 + 3 * 4.0):
            self.add_page()
        else:
            self.ln(SECTION_GAP)
        self.set_font("Helvetica", "B", 11)
        self.multi_cell(0, 5, text)
        self.ln(1.0)
        self.set_font("Helvetica", size=9)

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", size=9)
        self.multi_cell(0, 4.0, text)
        self.ln(PARAGRAPH_GAP)


def _fmt_far(value: float) -> str:
    """False-alarm rates are small; three decimals would round T2 to 0.001."""
    return "0" if value == 0 else f"{value:.5f}"


def _table(pdf: Report, detection: pd.DataFrame) -> None:
    """False-alarm row, then detection rate and median delay for faults 1 to 20.

    Widths are fractions of the frame width. A fixed millimetre sum wider than
    the margins pushes the cursor past the right edge, and the next paragraph
    then has no room to draw a character.
    """
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0,
        4.0,
        "Table 1. Detection rate and median delay per fault and statistic. Fault 0 is the "
        "false-alarm rate on fault-free test runs 401-500.",
    )
    pdf.set_x(pdf.l_margin)
    shares = (0.08, 0.12, 0.18, 0.12, 0.18, 0.13, 0.19)
    widths = [share * pdf.epw for share in shares]
    headers = ["Fault", "T2 rate", "T2 delay", "SPE rate", "SPE delay", "Ridge rate", "Ridge delay"]
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_fill_color(230, 236, 242)
    for text, width in zip(headers, widths):
        pdf.cell(width, 4.2, text, border=0, fill=True)
    pdf.ln(4.2)
    pdf.set_font("Helvetica", size=7.5)
    far_cells = ["0"]
    for detector in ("T2", "SPE", "ridge"):
        far_cells += [_fmt_far(_rate(detection, 0, detector)), "-"]
    for text, width in zip(far_cells, widths):
        pdf.cell(width, 3.8, text, border=0)
    pdf.ln(3.8)
    for fault in range(1, 21):
        fill = fault % 2 == 0
        if fill:
            pdf.set_fill_color(248, 248, 248)
        cells = [
            str(fault),
            _fmt_rate(_rate(detection, fault, "T2")),
            _fmt_delay(detection, fault, "T2"),
            _fmt_rate(_rate(detection, fault, "SPE")),
            _fmt_delay(detection, fault, "SPE"),
            _fmt_rate(_rate(detection, fault, "ridge")),
            _fmt_delay(detection, fault, "ridge"),
        ]
        for text, width in zip(cells, widths):
            pdf.cell(width, 3.8, text, border=0, fill=fill)
        pdf.ln(3.8)
    pdf.ln(PARAGRAPH_GAP)


def _late_fraction(frame: pd.DataFrame, column: str, threshold: float) -> tuple[float, float]:
    """Share of samples over the threshold just after the fault, and late in the run."""
    early = frame[(frame["sample"] > 20) & (frame["sample"] <= 40)]
    late = frame[frame["sample"] >= 200]
    early_rate = float((early[column] > threshold).mean()) if len(early) else 0.0
    late_rate = float((late[column] > threshold).mean()) if len(late) else 0.0
    return early_rate, late_rate


def _test_crossings(frame: pd.DataFrame, column: str, threshold: float) -> tuple[int, int, int]:
    """Test-run rows, rows over the line, and rows in alarm under the three-sample rule."""
    test = frame[(frame["faultNumber"] == 0) & frame["simulationRun"].between(401, 500)]
    alarmed = attach_alarms(test, column, threshold)
    return len(test), int((test[column] > threshold).sum()), int(alarmed["alarm"].sum())


def _first_alarms(
    frame: pd.DataFrame, column: str, threshold: float, fault: int
) -> set[tuple[int, int]]:
    """(run, first alarmed sample after 20) for every run of one fault that alarmed."""
    part = attach_alarms(frame[frame["faultNumber"] == fault], column, threshold)
    hits = part[part["alarm"] & (part["sample"] > 20)]
    return {(int(run), int(sample)) for run, sample in hits.groupby("simulationRun")["sample"].min().items()}


def _describe(faults: list[int]) -> str:
    if not faults:
        return "none"
    return ", ".join(f"{fault} ({FAULT_DESCRIPTION[fault].split(';')[0]})" for fault in faults)


def write_report(
    detection: pd.DataFrame,
    thresholds: pd.DataFrame,
    contributions: pd.DataFrame,
    figure: Path,
    k: int,
    variance: float,
    pca_scores: pd.DataFrame,
    ridge_scores: pd.DataFrame,
    destination: Path | None = None,
) -> Path:
    """Write REPORT.pdf at the project root."""
    out = destination or (ROOT / "REPORT.pdf")
    threshold_map = dict(zip(thresholds["detector"], thresholds["threshold"]))
    plot_fault = pick_plot_fault(detection)
    diagnosis_faults = pick_diagnosis_faults(contributions)

    caught = {name: _caught(detection, name) for name in ("T2", "SPE", "ridge")}
    missed_all = [
        fault
        for fault in range(1, 21)
        if fault not in caught["T2"]
        and fault not in caught["SPE"]
        and fault not in caught["ridge"]
    ]
    far = {name: _rate(detection, 0, name) for name in ("T2", "SPE", "ridge")}

    # Section 3 and 7: how the three-sample rule changes the test-run counts.
    score_frames = {"T2": (pca_scores, "T2"), "SPE": (pca_scores, "SPE"), "ridge": (ridge_scores, "score")}
    crossings = {
        name: _test_crossings(frame, column, float(threshold_map[name]))
        for name, (frame, column) in score_frames.items()
    }
    clear_all = [
        fault for fault in range(1, 21)
        if all(_rate(detection, fault, name) >= 0.50 for name in ("T2", "SPE", "ridge"))
    ]
    ridge_best = sum(
        _rate(detection, fault, "ridge") >= max(_rate(detection, fault, "T2"), _rate(detection, fault, "SPE"))
        for fault in range(1, 21)
    )
    fastest = float(detection.loc[detection["fault"] > 0, "median_delay_min"].min())
    fault16_t2_delay = float(
        detection.loc[(detection["fault"] == 16) & (detection["detector"] == "T2"), "median_delay_min"].iloc[0]
    )
    fault19_max = max(_rate(detection, 19, name) for name in ("T2", "SPE", "ridge"))
    t2_hidden = {
        fault: _first_alarms(pca_scores, "T2", float(threshold_map["T2"]), fault) for fault in (3, 9, 15)
    }
    shared_spike = (
        len(t2_hidden[3]) == 1 and t2_hidden[3] == t2_hidden[9] == t2_hidden[15]
    )

    plot_run = pca_scores[
        (pca_scores["faultNumber"] == plot_fault) & (pca_scores["simulationRun"] == 1)
    ]
    ridge_run = ridge_scores[
        (ridge_scores["faultNumber"] == plot_fault) & (ridge_scores["simulationRun"] == 1)
    ]
    t2_early, t2_late = _late_fraction(plot_run, "T2", float(threshold_map["T2"]))
    spe_early, spe_late = _late_fraction(plot_run, "SPE", float(threshold_map["SPE"]))
    ridge_early, ridge_late = _late_fraction(ridge_run, "score", float(threshold_map["ridge"]))

    def _windows(fault: int) -> dict[str, tuple[float, float]]:
        """Early (samples 21-40) and late (samples 200-500) exceedance on run 1."""
        pca_part = pca_scores[
            (pca_scores["faultNumber"] == fault) & (pca_scores["simulationRun"] == 1)
        ]
        ridge_part = ridge_scores[
            (ridge_scores["faultNumber"] == fault) & (ridge_scores["simulationRun"] == 1)
        ]
        return {
            "T2": _late_fraction(pca_part, "T2", float(threshold_map["T2"])),
            "SPE": _late_fraction(pca_part, "SPE", float(threshold_map["SPE"])),
            "ridge": _late_fraction(ridge_part, "score", float(threshold_map["ridge"])),
        }

    fault4 = _windows(4)
    fault5 = _windows(5)

    pdf = Report()
    pdf.add_page()
    pdf.h1("Detecting Tennessee Eastman faults from normal operation only")
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(
        0,
        4.15,
        "06-763 miniproject, team banana_bread_matcha_latte. PCA keeps "
        f"k = {k} components ({variance:.1%} of the training variance). Thresholds are "
        "the 99th percentile of validation runs 301-400.",
    )
    pdf.ln(PARAGRAPH_GAP)

    pdf.h2("1. The plant and the task")
    pdf.body(
        "The Tennessee Eastman Process is a simulated chemical plant with a reactor, "
        "condenser, vapour-liquid separator, recycle compressor and product stripper. "
        "Fifty-two signals are recorded every 3 minutes: 41 measurements (xmeas_1 to "
        "xmeas_41) and 11 manipulated variables (xmv_1 to xmv_11). A run is 500 samples, "
        "or 25 hours. In a faulty run the fault starts after sample 20, so samples 1-20 "
        "are normal and samples 21-500 are faulty."
    )
    pdf.body(
        "The task is to detect 20 fault types with models trained on normal operation "
        "only, because a classifier trained on labelled faults cannot recognise a fault "
        "type it has not seen. We build two detectors: a PCA monitor (T2 and SPE) and a "
        "ridge forecast-residual monitor. Both are fitted on fault-free runs 1-300. "
        "Thresholds are set on fault-free runs 301-400 and false alarms are measured on "
        "fault-free runs 401-500. Faults 1-20, 20 runs each, are scored afterwards. No "
        "faulty sample is used to fit a model or set a threshold."
    )

    pdf.h2("2. The two detectors")
    pdf.body(
        "PCA monitor. Each channel is standardized with the training mean and standard "
        "deviation (ddof = 1). PCA keeps the smallest number of components whose "
        f"eigenvalues sum to at least 90% of the total: k = {k} ({variance:.1%}). With "
        "loadings P and scores t = P'z, T2 is the sum of t_i^2 / lambda_i over the retained "
        "components. It is large when a sample moves an unusual distance along the normal "
        "directions of variation. SPE is the squared reconstruction error |z - PP'z|^2. "
        "It is large when the channels stop following their normal correlations."
    )
    pdf.body(
        "Forecast monitor. This is the Lecture 8 one-step residual detector applied to all "
        "52 channels. The features are the standardized samples at t-1 and t-2 (104 "
        "columns) and the target is the sample at t. Lags never cross a run boundary, so "
        "samples 1 and 2 of each run have no score. One Ridge(alpha = 1) is fitted on the "
        "training runs to predict all 52 channels. Each residual is divided by that "
        "channel's training residual standard deviation (ddof = 1), and the score is the "
        "sum of squares. It is large when the last two samples do not predict the current one."
    )
    pdf.body(
        "Thresholds and alarms. Each threshold is numpy.quantile of the validation scores "
        f"at 0.99 with the default interpolation: T2 {float(threshold_map['T2']):.4g}, "
        f"SPE {float(threshold_map['SPE']):.4g}, ridge {float(threshold_map['ridge']):.4g}. "
        "A sample is in alarm when it and the two preceding samples of the same run all "
        f"exceed the threshold. Figure 1 shows fault {plot_fault}, run 1 "
        f"({FAULT_DESCRIPTION[plot_fault]}). The share of samples above the threshold in "
        f"samples 21-40, then 200-500, is T2 {t2_early:.2f} then {t2_late:.2f}, SPE "
        f"{spe_early:.2f} then {spe_late:.2f}, and ridge {ridge_early:.2f} then "
        f"{ridge_late:.2f}. The dashed line marks sample 20 and the horizontal line is "
        "the threshold."
    )

    pdf.set_x(pdf.l_margin)
    pdf.image(str(figure), w=168)
    pdf.ln(PARAGRAPH_GAP)

    pdf.h2("3. Results")
    pdf.body(
        "Table 1 gives the detection rate and median delay for each fault and statistic. "
        "The detection rate is the share of samples after sample 20 in alarm, averaged "
        "over the 20 runs. The delay is the median, over runs that alarmed, of the minutes "
        'from sample 20 to the first alarm. "none" means all 20 runs were missed, and "N '
        'missed" counts runs with no alarm. Fault 0 is the false-alarm rate, the share of '
        "samples in alarm on test runs 401-500."
    )
    _table(pdf, detection)
    pdf.body(
        f"False alarms. On the test runs, T2 exceeds its threshold on {crossings['T2'][1]} "
        f"of {crossings['T2'][0]:,} samples, SPE on {crossings['SPE'][1]} of "
        f"{crossings['SPE'][0]:,} and ridge on {crossings['ridge'][1]} of "
        f"{crossings['ridge'][0]:,}, close to the 1% a 99th percentile implies. After the "
        f"three-sample rule, {crossings['T2'][2]}, {crossings['SPE'][2]} and "
        f"{crossings['ridge'][2]} samples are in alarm. SPE and ridge exceedances are "
        "isolated. T2 exceedances cluster, because slow normal variation lies inside the "
        "PCA subspace and keeps T2 high for several samples."
    )
    pdf.body(
        "Detection. All three statistics reach a rate of 0.50 or more on faults "
        f"{', '.join(str(fault) for fault in clear_all)}, and all three stay below "
        f"{CAUGHT_RATE:.2f} on faults {', '.join(str(fault) for fault in missed_all)}. "
        f"Ridge has the highest rate, or ties for it, on {ridge_best} of 20 faults. The "
        f"shortest median delay is {fastest:.0f} minutes, because the first three-sample "
        "window entirely after the fault ends at sample 23. An earlier alarm needs a "
        "pre-fault sample above the threshold."
    )
    hidden_text = ""
    if shared_spike:
        run, sample = next(iter(t2_hidden[3]))
        hidden_text = (
            f" T2's only alarms on faults 3, 9 and 15 start in run {run} at sample {sample}, "
            "in all three faults. The three T2 traces are nearly identical around that "
            "sample, which suggests that runs with the same number share their background "
            "noise. The alarm is a normal excursion, not a detection."
        )
    pdf.body(
        "A delay means at least one run alarmed, not that the fault was detected. T2 on "
        f"fault 16 has a median delay of {fault16_t2_delay:.0f} minutes at a rate of "
        f"{_rate(detection, 16, 'T2'):.3f}, and fault 19 has delays on all three "
        f"statistics at rates of {fault19_max:.3f} or less." + hidden_text
    )

    pdf.h2("4. Comparison")
    ridge_only = [
        fault for fault in range(1, 21)
        if _rate(detection, fault, "ridge") >= 0.50 and _rate(detection, fault, "SPE") < 0.50
    ]
    pdf.body(
        "T2 responds to large moves along normal directions, SPE to broken correlations "
        "between channels, and ridge to samples that the previous two do not predict. "
        f"Taking 0.50 as a clear catch and {CAUGHT_RATE:.2f} as a miss, ridge clearly "
        "catches every fault that SPE clearly catches, and also catches "
        f"{_describe(ridge_only)}, which SPE does not. SPE clearly catches fault 4 and T2 "
        f"does not (T2 {_rate(detection, 4, 'T2'):.3f}, SPE {_rate(detection, 4, 'SPE'):.3f}, "
        f"ridge {_rate(detection, 4, 'ridge'):.3f}). No fault is clearly caught by T2 and "
        "missed by SPE."
    )
    pdf.body(
        "Fault 4 is a step in reactor cooling-water inlet temperature. On run 1, the share "
        "of samples above the threshold in samples 21-40, then 200-500, is "
        f"T2 {fault4['T2'][0]:.2f} then {fault4['T2'][1]:.2f}, "
        f"SPE {fault4['SPE'][0]:.2f} then {fault4['SPE'][1]:.2f}, and "
        f"ridge {fault4['ridge'][0]:.2f} then {fault4['ridge'][1]:.2f}. The temperature "
        "controller moves the reactor cooling-water valve and returns the measurements "
        f"toward the normal region, so T2 alarms only briefly (rate "
        f"{_rate(detection, 4, 'T2'):.3f}, although every run alarms at least once). The "
        "valve and the reactor temperature no longer follow their normal relationship, so "
        "SPE and the ridge residual stay high."
    )
    pdf.body(
        "Fault 5 is a step in condenser cooling-water inlet temperature, and the detectors "
        "disagree most here. On run 1 all three exceed the threshold just after the fault "
        f"(T2 {fault5['T2'][0]:.2f}, SPE {fault5['SPE'][0]:.2f}, ridge {fault5['ridge'][0]:.2f}). "
        f"Late in the run T2 is at {fault5['T2'][1]:.2f} and SPE at {fault5['SPE'][1]:.2f}, "
        f"while ridge stays at {fault5['ridge'][1]:.2f}. The controllers return the "
        "operating point inside both PCA limits, but the one-step forecast stays wrong. "
        f"Over the 20 runs the rates are T2 {_rate(detection, 5, 'T2'):.3f}, "
        f"SPE {_rate(detection, 5, 'SPE'):.3f} and ridge {_rate(detection, 5, 'ridge'):.3f}. "
        f"Fault {plot_fault} in Figure 1 is the opposite case: the disturbance is not "
        f"absorbed and the late shares stay high (T2 {t2_late:.2f}, SPE {spe_late:.2f}, "
        f"ridge {ridge_late:.2f})."
    )

    pdf.h2("5. Faults nobody catches")
    missed_text = _describe(missed_all) if missed_all else "none"

    def _triple(fault: int, value) -> str:
        return " / ".join(str(value(detection, fault, name)) for name in ("T2", "SPE", "ridge"))

    def _rates(fault: int) -> str:
        return " / ".join(f"{_rate(detection, fault, name):.3f}" for name in ("T2", "SPE", "ridge"))

    spike_note = (
        " The single T2 alarm on each is the run-7 excursion described in Section 3."
        if shared_spike else ""
    )
    pdf.body(
        f"All three statistics stay below {CAUGHT_RATE:.2f} on {missed_text}. Faults 3, 9 "
        "and 15 are the faults the project brief and Russell, Chiang and Braatz (2000) "
        "single out as hard to detect. Fault 3 is a step in D-feed temperature, fault 9 is "
        "random variation in the same temperature, and fault 15 is a sticking condenser "
        "cooling-water valve. Rates (T2 / SPE / ridge) are "
        f"{_rates(3)} for fault 3, {_rates(9)} for fault 9 and {_rates(15)} for fault 15. "
        f"Runs missed are {_triple(3, _missed)}, {_triple(9, _missed)} and "
        f"{_triple(15, _missed)}." + spike_note
    )
    pdf.body(
        "These faults either change the recorded channels too little or are cancelled by "
        "the control loops before they reach them, so the samples stay within the "
        "variation of the 300 normal training runs. Missing them reflects the sensors and "
        "the controllers, not a healthy plant. Fault 19 is different. Its rates are low, "
        f"but runs missed are only {_triple(19, _missed)}, so it is weak rather than "
        "invisible."
    )

    pdf.h2("6. Diagnosis")
    pdf.body(
        "SPE and the ridge score are sums of squared terms, one per channel. For each "
        "fault, those terms are averaged over every sample in alarm after sample 20 across "
        "the 20 runs, and the five largest are kept. A contribution shows where a fault "
        "appears, not necessarily where it starts: Westerhuis, Gurden and Smilde (2000) "
        "show that contributions smear onto correlated channels."
    )
    notes = {
        1: (
            "SPE ranks the stream-4 flow and its valve highest, where the fault enters. "
            "Ridge ranks the A feed and its valve highest, where the controllers correct "
            "the ratio. Product composition E and the reactor cooling-water outlet "
            "temperature are downstream effects."
        ),
        4: (
            "The inlet temperature is not measured. Both detectors rank the reactor "
            "cooling-water valve first and the reactor temperature second, which is the "
            "loop that responds to the disturbance. The remaining SPE terms are small, so "
            "the ranking is not smeared."
        ),
        6: (
            "Ridge ranks the A feed and its valve first, the stream that was lost. SPE "
            "ranks the compressor recycle valve first, then reactor cooling and compressor "
            "work, with the A feed fifth. SPE shows the plant-wide correction: losing the "
            "A feed changes the recycle flow and the reactor heat balance, which matches "
            "the smearing described by Westerhuis, Gurden and Smilde."
        ),
    }
    if not diagnosis_faults:
        pdf.body("No fault has an alarmed sample after sample 20, so there is no ranking.")
    for fault in diagnosis_faults:
        spe_channels = _top_channels(contributions, fault, "SPE")
        ridge_channels = _top_channels(contributions, fault, "ridge")
        pdf.body(
            f"Fault {fault}: {FAULT_DESCRIPTION[fault]}. "
            f"SPE top five: {_names(spe_channels)}. "
            f"Ridge top five: {_names(ridge_channels)}. "
            + notes.get(
                fault,
                "Channels on the disturbed equipment support the fault description. The "
                "others show where the plant moved after the controllers reacted.",
            )
        )

    pdf.h2("7. Limits")
    raw_per_day = {
        name: SAMPLES_PER_DAY * crossings[name][1] / crossings[name][0] for name in crossings
    }
    alarm_per_day = {
        name: SAMPLES_PER_DAY * crossings[name][2] / crossings[name][0] for name in crossings
    }
    pdf.body(
        "Cut-offs. The 0.10 line for a catch and the 0.50 line for a clear catch are our "
        "own choices, not the project's, and they decide which faults Sections 4 and 5 "
        f"call missed. Fault 19, at {fault19_max:.3f} or less on every statistic, would "
        "change category with a lower line. The thresholds are empirical percentiles, not "
        "the textbook F and Jackson-Mudholkar limits, which assume independent normal samples."
    )
    pdf.body(
        f"Alarm rule. At {SAMPLES_PER_DAY} samples a day, raw exceedances on the test runs "
        f"occur about {raw_per_day['T2']:.1f}, {raw_per_day['SPE']:.1f} and "
        f"{raw_per_day['ridge']:.1f} times a day for T2, SPE and ridge, close to the 4.8 a "
        "day reported in Lecture 8 for a 99th-percentile threshold. The three-sample rule "
        f"cuts alarmed samples to {alarm_per_day['T2']:.2f}, {alarm_per_day['SPE']:.2f} "
        f"and {alarm_per_day['ridge']:.2f} a day, at the cost of at least "
        f"{2 * SAMPLE_MINUTES} extra minutes before any alarm. A real plant would choose "
        "this rule by weighing the cost of a false alarm against the cost of a late one."
    )
    pdf.body(
        f"Sample size. Zero alarmed samples out of {crossings['SPE'][0]:,} in 100 test runs "
        "shows the false-alarm rate is small, not zero, and correlated neighbouring samples "
        "make the evidence weaker than the count suggests. Each fault has only 20 runs, and "
        "runs with the same number appear to share background noise across faults, so "
        "differences between faults rest on fewer independent draws than the table "
        "suggests. Every metric also assumes a known fault start after sample 20. A real "
        "plant does not know when a fault began, and operators care about alarm events and "
        "time to first alarm more than the share of samples in alarm."
    )
    pdf.body(
        "Scope. The data are one simulated operating mode with scripted faults. They "
        "contain no drift in normal operation, no failed sensors and no faults during "
        "setpoint changes. Contribution ranks are not root causes. The ridge model is "
        "linear with two lags, so a slow oscillation can look predictable. A fault that "
        "the controllers fully absorb is invisible to all three statistics, whatever the "
        "threshold."
    )

    pdf.h2("8. AI use")
    pdf.body(
        "Cursor's coding agent (Grok) drafted the pipeline and the first version of this "
        "report, constrained to follow the project recipe: runs 1-300 for fitting, "
        "301-400 for thresholds and 401-500 for false alarms, ddof = 1 standardization, "
        "a 90% PCA cutoff, Ridge(alpha = 1) on lags t-1 and t-2, 99th-percentile "
        "thresholds, the three-sample alarm rule and top-five contributions. Claude Code "
        "(Anthropic) was used to review the evaluation code and to revise the text of "
        "this report. Every number in the report is computed from the score files, not "
        "typed in, and the PDF was not edited by hand. The team checked the code against "
        "the recipe. The course evidence script, run on 8 October 2026, passed all "
        "automatic checks (10/10). Each member is responsible for explaining the code "
        "they own and the comparison in Section 4."
    )

    # The appendix is outside the four-page body. Start a new page so the
    # break is visible. page_no() here is the last body page.
    print(f"body_pages={pdf.page_no()}")
    pdf.add_page()
    pdf.h2("Appendix: contributions")
    pdf.body("This appendix does not count toward the four-page limit. One entry per member.")
    for person in TEAM:
        pdf.body(
            f"{person['name']} ({person['andrew']}). {person['role']}. "
            f"Code: {person['code']}."
        )

    pdf.output(str(out))
    print(f"report  {out.name}  pages={pdf.pages_count}")
    return out
