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
)


# One entry per member. Andrew IDs are not stored in this repository.
# Replace the placeholder and add the other three members before submitting.
TEAM = [
    {
        "name": "Sean",
        "andrew": "ANDREW_ID",
        "email": "schao1215@gmail.com",
        "work": (
            "Shared preprocessing, PCA monitor, ridge forecast-residual monitor, "
            "thresholds, alarm rule, detection table, contribution table, figures, "
            "and this report. The pipeline was drafted with Cursor's coding agent "
            "and checked against the project recipe."
        ),
    }
]


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
        self.ln(1.0)
        self.set_font("Helvetica", "B", 11)
        self.multi_cell(0, 5, text)
        self.set_font("Helvetica", size=9)

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", size=9)
        self.multi_cell(0, 4.0, text)
        self.ln(0.6)

    def quote(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 4.15, text)
        self.set_text_color(0, 0, 0)
        self.ln(0.6)


def _table(pdf: Report, detection: pd.DataFrame) -> None:
    """Detection rate and median delay for faults 1 to 20.

    Widths are fractions of the frame width. A fixed millimetre sum wider than
    the margins pushes the cursor past the right edge, and the next paragraph
    then has no room to draw a character.
    """
    shares = (0.08, 0.12, 0.18, 0.12, 0.18, 0.13, 0.19)
    widths = [share * pdf.epw for share in shares]
    headers = ["Fault", "T2 rate", "T2 delay", "SPE rate", "SPE delay", "Ridge rate", "Ridge delay"]
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_fill_color(230, 236, 242)
    for text, width in zip(headers, widths):
        pdf.cell(width, 4.2, text, border=0, fill=True)
    pdf.ln(4.2)
    pdf.set_font("Helvetica", size=7.5)
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
    pdf.ln(1)


def _late_fraction(frame: pd.DataFrame, column: str, threshold: float) -> tuple[float, float]:
    """Share of samples over the threshold just after the fault, and late in the run."""
    early = frame[(frame["sample"] > 20) & (frame["sample"] <= 40)]
    late = frame[frame["sample"] >= 200]
    early_rate = float((early[column] > threshold).mean()) if len(early) else 0.0
    late_rate = float((late[column] > threshold).mean()) if len(late) else 0.0
    return early_rate, late_rate


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
        "Miniproject. PCA retained "
        f"k = {k} components ({variance:.1%} of the training variance). "
        "Thresholds are the empirical 99th percentile of validation runs 301-400.",
    )
    pdf.ln(1)

    pdf.h2("1. The plant and the task")
    pdf.body(
        "The Tennessee Eastman Process is a simulated chemical plant. Gaseous reactants "
        "enter an exothermic reactor, the product is condensed and recycled, and a stripper "
        "sends the liquid product downstream. Fifty-two signals are recorded every three "
        "minutes: forty-one measurements and eleven valve positions. A run is 500 samples, "
        "which is 25 hours. In a faulty run the disturbance starts at one hour, so samples "
        "1 to 20 are still normal and samples 21 to 500 are not."
    )
    pdf.body(
        "Faults are uncommon, and the next one is often a kind nobody has labelled. A "
        "classifier trained on old fault examples cannot see that new kind. Process "
        "monitoring therefore learns what normal operation looks like, and raises an alarm "
        "when a new sample stops looking normal. This project builds two such detectors. "
        "Both are trained only on fault-free runs 1 to 300. Thresholds use runs 301 to 400. "
        "False alarms use runs 401 to 500. The same twenty faults are then scored. No faulty "
        "row is used to fit a model or to choose a threshold."
    )

    pdf.h2("2. The two detectors")
    pdf.body(
        f"The PCA monitor standardizes each channel by the training mean and the training "
        f"standard deviation, using divisor n - 1, then keeps the smallest set of principal "
        f"components whose eigenvalues sum to at least 90 percent of the total. That cutoff "
        f"is k = {k} ({variance:.1%}). For a standardized row z and loadings P, the score "
        "vector is t = P transpose z. T2 sums t_i squared over lambda_i for the retained "
        "components. It is large when the plant still moves along its usual directions, but "
        "by an unusual amount. SPE is the squared length of the residual z - P P transpose z. "
        "It is large when the channels stop moving together, so the retained components "
        "cannot reconstruct the row."
    )
    pdf.quote(
        'Project primer: "T2 is the squared distance from the centre in that subspace, '
        'with each direction scaled by its own variance." "A large SPE means the channels '
        'have stopped moving together the way they normally do."'
    )
    pdf.body(
        "The forecast monitor is the Lecture 8 residual detector applied to all 52 channels. "
        "Inside each run the features are the standardized rows at t-1 and at t-2, 104 "
        "columns, and the target is the standardized row at t. One ridge regression with "
        "alpha = 1 is fit on the training runs and predicts all 52 channels at once. Each "
        "residual is divided by that channel's training residual standard deviation, again "
        "with divisor n - 1, then squared and summed. The first two samples of a run have "
        "no score, because t-2 does not exist, and a lag never crosses a run boundary. "
        "This score is large when the recent past does not predict the present. That is a "
        "different question from asking whether the operating point is far from the normal cloud."
    )
    pdf.body(
        "Each threshold is numpy.quantile of the validation scores at probability 0.99, "
        "with NumPy's default interpolation. "
        f"T2 is {float(threshold_map['T2']):.4g}, SPE is {float(threshold_map['SPE']):.4g}, "
        f"and the ridge score is {float(threshold_map['ridge']):.4g}. A sample is in alarm "
        "only when it and the two samples before it all exceed that line. "
        f"Figure 1 is fault {plot_fault}, run 1 ({FAULT_DESCRIPTION[plot_fault]}). "
        "On that run, the share of samples above the line in samples 21-40, then in "
        f"samples 200-500, is T2 {t2_early:.2f} then {t2_late:.2f}, "
        f"SPE {spe_early:.2f} then {spe_late:.2f}, "
        f"and ridge {ridge_early:.2f} then {ridge_late:.2f}. "
        "The dashed line is the last normal sample. The horizontal line is the "
        "validation 99th percentile of that statistic."
    )

    pdf.set_x(pdf.l_margin)
    pdf.image(str(figure), w=168)
    pdf.ln(1)

    pdf.h2("3. Results")
    pdf.body(
        "The false-alarm rate is the share of samples in alarm on fault-free test runs "
        f"401 to 500. T2 {far['T2']:.4f}, SPE {far['SPE']:.4f}, ridge {far['ridge']:.4f}. "
        "The detection rate, for each faulty run, is the share of samples after sample 20 "
        "that are in alarm, averaged over the 20 runs. Delay is the median, over the runs "
        "that did alarm, of the minutes from sample 20 to the first later alarm. Samples "
        "are 3 minutes apart, so an alarm at sample 21 is a delay of 3 minutes. The cell "
        '"none" means every run was missed. A count after the delay is how many of the '
        "20 runs never alarmed."
    )
    _table(pdf, detection)

    pdf.h2("4. Comparison")
    pdf.body(
        "T2 is large inside the normal subspace, and a quiet direction counts more because "
        "its eigenvalue is in the denominator. SPE is large when channels stop moving "
        "together. The ridge score is large when the last two samples fail to predict the "
        "present. A rate of at least 0.50 is a clear catch; a rate below "
        f"{CAUGHT_RATE:.2f} is a miss. On that 0.50 line, every fault SPE catches, ridge "
        f"catches too. Ridge also clearly catches {_describe([fault for fault in range(1, 21) if _rate(detection, fault, 'ridge') >= 0.50 and _rate(detection, fault, 'SPE') < 0.50])}. "
        f"SPE clearly catches fault 4 and T2 does not: T2 {_rate(detection, 4, 'T2'):.3f}, "
        f"SPE {_rate(detection, 4, 'SPE'):.3f}, ridge {_rate(detection, 4, 'ridge'):.3f}. "
        f"No fault is clearly caught by T2 and missed by SPE. "
        f"The faults below {CAUGHT_RATE:.2f} on every statistic are {_describe(missed_all)}."
    )
    pdf.body(
        "Fault 4 is a step in reactor cooling-water inlet temperature. On run 1, the share "
        "of samples above the line in samples 21-40, then in samples 200-500, is "
        f"T2 {fault4['T2'][0]:.2f} then {fault4['T2'][1]:.2f}, "
        f"SPE {fault4['SPE'][0]:.2f} then {fault4['SPE'][1]:.2f}, "
        f"ridge {fault4['ridge'][0]:.2f} then {fault4['ridge'][1]:.2f}. "
        "The temperature loop moves the cooling-water valve and pulls the measurements "
        "back toward the normal cloud, so T2 does not stay in alarm (rate "
        f"{_rate(detection, 4, 'T2'):.3f}, though every run does alarm at least once). "
        "SPE stays large because that valve and the reactor temperature are no longer on "
        "their normal relationship. The forecast was trained on the normal relationship, "
        "so its residual stays large as well."
    )
    pdf.body(
        "Fault 5 is a step in condenser cooling-water inlet temperature, and it is the "
        "case where the detectors disagree. Just after the fault on run 1, all three are "
        f"over the line (T2 {fault5['T2'][0]:.2f}, SPE {fault5['SPE'][0]:.2f}, "
        f"ridge {fault5['ridge'][0]:.2f}). Late in that run T2 is {fault5['T2'][1]:.2f} "
        f"and SPE is {fault5['SPE'][1]:.2f}, while ridge is still {fault5['ridge'][1]:.2f}. "
        "The controllers bring the static operating point back inside both PCA limits. "
        "The one-step forecast does not recover. Averaged over the 20 runs the rates are "
        f"T2 {_rate(detection, 5, 'T2'):.3f}, SPE {_rate(detection, 5, 'SPE'):.3f}, "
        f"ridge {_rate(detection, 5, 'ridge'):.3f}. "
        f"Fault {plot_fault} in Figure 1 is the opposite pattern: the disturbance is not "
        f"absorbed, and the late shares stay high "
        f"(T2 {t2_late:.2f}, SPE {spe_late:.2f}, ridge {ridge_late:.2f})."
    )

    pdf.h2("5. Faults nobody catches")
    missed_text = _describe(missed_all) if missed_all else (
        "no fault sits below the 0.10 line on all three statistics"
    )
    pdf.body(
        f"Faults missed by all three statistics, under that same 0.10 line: {missed_text}. "
        "The project check, and the Tennessee Eastman studies it points to, single out "
        "faults 3, 9, and 15. Fault 3 is a step in D-feed temperature, fault 9 is random "
        "variation of that same temperature, and fault 15 is a sticking condenser "
        "cooling-water valve. Monitoring papers call these three unobservable: on the 52 "
        "recorded channels they look like normal operation. The temperature disturbance "
        "barely appears in the measurements, and the level and temperature loops can "
        "absorb the sticking valve. Our rates are "
        f"fault 3: T2 {_rate(detection, 3, 'T2'):.3f}, SPE {_rate(detection, 3, 'SPE'):.3f}, "
        f"ridge {_rate(detection, 3, 'ridge'):.3f}; "
        f"fault 9: T2 {_rate(detection, 9, 'T2'):.3f}, SPE {_rate(detection, 9, 'SPE'):.3f}, "
        f"ridge {_rate(detection, 9, 'ridge'):.3f}; "
        f"fault 15: T2 {_rate(detection, 15, 'T2'):.3f}, SPE {_rate(detection, 15, 'SPE'):.3f}, "
        f"ridge {_rate(detection, 15, 'ridge'):.3f}. "
        "Runs missed, written as T2 / SPE / ridge, are "
        f"fault 3: {_missed(detection, 3, 'T2')}/{_missed(detection, 3, 'SPE')}/{_missed(detection, 3, 'ridge')}, "
        f"fault 9: {_missed(detection, 9, 'T2')}/{_missed(detection, 9, 'SPE')}/{_missed(detection, 9, 'ridge')}, "
        f"fault 15: {_missed(detection, 15, 'T2')}/{_missed(detection, 15, 'SPE')}/{_missed(detection, 15, 'ridge')}."
    )
    pdf.body(
        "That says something about these sensors and this controller, not that the plant "
        "is healthy. A disturbance can be real and still be invisible when it is rejected "
        "before it reaches a recorded channel, or when it only rearranges the plant inside "
        "the variation already present in the 300 normal runs."
    )

    pdf.h2("6. Diagnosis")
    pdf.body(
        "SPE and the ridge score are sums of squares, so each splits into one term per "
        "channel. For each fault those terms are averaged over every sample that is in "
        "alarm after sample 20, across the 20 runs, and the top five channels are kept. "
        "The average shows where the fault appears. Westerhuis, Gurden and Smilde (2000) "
        "show that contribution plots smear: a channel correlated with the disturbed one "
        "inherits part of the score, so the largest term is not always the place the "
        "fault started."
    )
    notes = {
        1: (
            "The fault is an A/C feed-ratio step in stream 4. SPE's two largest terms "
            "are the stream-4 flow and its valve, which is where the fault is introduced. "
            "Ridge instead puts almost all of its score on the A feed and the A-feed valve: "
            "the ratio change shows up there as a correction. Product composition E and the "
            "reactor cooling-water outlet temperature in the SPE list are downstream of that "
            "feed change. The detectors agree that a feed moved, and they disagree which leg is loudest."
        ),
        4: (
            "The inlet temperature itself is not one of the 52 channels. Both detectors "
            "rank the reactor cooling-water valve first and the reactor temperature second, "
            "which is the loop that answers the disturbance. The other SPE terms are much "
            "smaller, so this ranking is not smeared across the plant."
        ),
        6: (
            "Ridge ranks the A feed and its valve first, which is the stream that was lost. "
            "SPE's largest term is the compressor recycle valve, then reactor cooling and "
            "compressor work; the A feed itself is only fifth. The plot is showing the "
            "plant-wide correction, not only the stream that failed. That is the smearing "
            "Westerhuis, Gurden and Smilde describe, and it is also a real material balance: "
            "losing the A feed forces the recycle and the reactor heat balance to move."
        ),
    }
    if not diagnosis_faults:
        pdf.body(
            "No fault produced an alarming sample after sample 20, so there is no ranking."
        )
    for fault in diagnosis_faults:
        spe_channels = _top_channels(contributions, fault, "SPE")
        ridge_channels = _top_channels(contributions, fault, "ridge")
        pdf.body(
            f"Fault {fault}: {FAULT_DESCRIPTION[fault]}. "
            f"SPE top five: {_names(spe_channels)}. "
            f"Ridge top five: {_names(ridge_channels)}. "
            + notes.get(
                fault,
                "Channels on the disturbed equipment support the fault description; "
                "the others are where the plant moved after the controllers reacted.",
            )
        )

    pdf.h2("7. Limits")
    pdf.body(
        "The threshold is an empirical percentile of held-out normal runs, not the "
        "textbook F limit for T2 or the Jackson-Mudholkar approximation for SPE. Those "
        "limits assume independent normal samples. Samples taken three minutes apart are "
        "neither, which is why the recipe uses one percentile for all three statistics. "
        "The number means that about one percent of normal validation rows sit above the "
        "line. The three-in-a-row rule then makes the test-set false-alarm rate smaller, "
        "which is what the fault-0 row reports."
    )
    pdf.body(
        "This setup cannot say what a real plant would do. The records are one simulated "
        "operating mode with scripted faults. They do not include a slow drift of the "
        "normal regime, a sensor that fails stuck, or a fault that arrives during a "
        "setpoint change. Contribution ranks are not root causes. The ridge model is "
        "linear and looks back only two samples, so a slow oscillation can look "
        "predictable. A fault absorbed by the controllers stays invisible to both "
        "detectors, however the threshold is set."
    )

    pdf.h2("8. AI use")
    pdf.body(
        "The pipeline and this report were drafted with Cursor's coding agent (Grok). "
        "The draft was required to follow the project recipe: training runs 1-300, "
        "validation runs 301-400 for thresholds only, test runs 401-500 for false alarms, "
        "standardization with divisor n - 1, a 90 percent PCA cutoff, Ridge with alpha 1 "
        "on lags t-1 and t-2, numpy.quantile at 0.99, the three-sample alarm rule, and "
        "the top five contributions on alarming samples after sample 20. The course "
        "evidence script was not run. The formulas and the filters were checked in the "
        "code against that recipe, and the table in section 3 is computed from the score "
        "files rather than typed in. The PDF was not edited by hand after it was written. "
        "Each author still has to be able to explain the code they are responsible for "
        "and the comparison in section 4."
    )

    # The appendix is outside the four-page body. Start a new page so the
    # break is visible. page_no() here is the last body page.
    print(f"body_pages={pdf.page_no()}")
    pdf.add_page()
    pdf.h2("Appendix: contributions")
    pdf.body(
        "This appendix does not count toward the four-page limit. One entry per member."
    )
    for person in TEAM:
        pdf.body(
            f"{person['name']} ({person['andrew']}, {person['email']}). {person['work']}"
        )
    pdf.body(
        "The team has four members. Add the other three Andrew IDs, and what each person "
        "built, before this file is submitted. The evidence script does not read this page."
    )

    pdf.output(str(out))
    print(f"report  {out.name}  pages={pdf.pages_count}")
    return out
