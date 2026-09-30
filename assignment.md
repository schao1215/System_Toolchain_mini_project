# Miniproject: Detecting faults in a chemical plant without examples of faults

**Released:** Lecture 10 (2026-09-28) · **Due:** Friday 2026-10-09 · **Teams:** 4, working in pairs in the first week · **Points:** 15 (10 from the evidence script, 5 for the report) · **Weight:** 20 % of the course grade

## Overview

A plant runs normally almost all the time. Faults are uncommon and diverse/varied, and the next faulty is
usually of a certain type nobody has recorded before. A fault detector trained on labelled examples of past
faults simply "cannot see" a new one. So process monitoring usually works the other way round: learn
what normal operation looks like, and raise an alarm when the plant stops looking like that.

In this project your team builds two such fault detectors for the Tennessee Eastman Process (TEP), trains
both predictors on fault-free data only, and tests them against twenty different faults. One detector is
the classical tool of process monitoring: principal component analysis (PCA) with the T² and squared prediction error (SPE) statistics. The other is the forecast-residual detector we've seen in
[Lecture 8](../lectures/l08/notes.md), extended from one channel to all 52 that exist. Then, you compare
them fault by fault and explain the differences.

Both detectors are specified below, so an evidence script can rebuild each one from
the data and check your numbers. Your effort goes into making the pieces fit together,
measuring everything, and explaining what the numbers actually say about the plant.

## Learning outcomes

- Train an anomaly detector on normal data regime, and set its threshold on separate normal data.
- Build a PCA monitoring model and compute the T² and SPE statistics for new observations.
- Build a multivariate forecast-residual detector and explain what it responds to.
- Report detection rate, detection delay and false-alarm rate per fault, and say which faults
  none of the detectors can see.
- Work as a team on a shared, reproducible pipeline, building parts in parallel and then
  combining them afterwards.

## The data

Two Parquet files, subsets of the TEP simulations originally published by
[Rieth, Amsel, Tran and Cook (2017)](https://doi.org/10.7910/DVN/6C3JR1). The
original faulty file needs about 9 GB of memory to read, so the we host only the part you need (details below).

| File                                                                                                                                   | Rows    | Contents                                                          |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------- | ----------------------------------------------------------------- |
| [`tep_fault_free_training.parquet`](https://kitchin-services.cheme.cmu.edu/f26-06763/data/tep_fault_free_training.parquet)             | 250,000 | fault-free, `simulationRun` 1 to 500, `sample` 1 to 500           |
| [`tep_faulty_training_runs01-20.parquet`](https://kitchin-services.cheme.cmu.edu/f26-06763/data/tep_faulty_training_runs01-20.parquet) | 200,000 | `faultNumber` 1 to 20, `simulationRun` 1 to 20, `sample` 1 to 500 |

Download both into a `data/` folder and check them against the published
[checksums](https://kitchin-services.cheme.cmu.edu/f26-06763/data/SHA256SUMS). The evidence
script refuses files that do not match.

```bash
mkdir -p data && cd data
for f in tep_fault_free_training.parquet tep_faulty_training_runs01-20.parquet SHA256SUMS; do
  curl -fLO https://kitchin-services.cheme.cmu.edu/f26-06763/data/$f
done
shasum -a 256 -c SHA256SUMS
```

Samples are 3 minutes apart, so a run is 25 hours. In the faulty file, each fault is introduced
one hour into the run, so samples 1 to 20 are normal and samples 21 onward are faulty. The
twenty faults (IDV 1 to 20) are listed in the header of the original simulation code,
[`teprob.f`](https://github.com/camaramm/tennessee-eastman-profBraatz/blob/master/teprob.f),
and in Table 1 of Chiang, Russell and Braatz (2000), linked under Resources.

## Teams and how to split the work

Teams have four members. We suggest splitting into two pairs for the first week: one pair
builds the PCA monitor, the other builds the forecast monitor. The two detectors are independent from each other, which means you can work in parallel, but they share the preprocessing below, so agree on
it on the first day and put it in one place that both pairs can use.

In the second week the team combines the two halves. The thresholds, the alarms, the detection
table, the contributions and the report all need scores from both detectors, so neither pair can
finish without the other. Plan for that: a pair whose scores arrive on the last day leaves the
team no time to evaluate them.

The split is a suggestion, not a rule. Feel free to work as you see fit within your assigned group. The evidence script checks the team's whole pipeline and
does not ask who built which part. You report that yourselves, in an appendix to the report.

## The "project recipe"

### Shared by everyone

| Choice          | Value                                                                                                            |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| Channels        | all 52: `xmeas_1` to `xmeas_41` and `xmv_1` to `xmv_11`                                                          |
| Training runs   | fault-free runs 1 to 300                                                                                         |
| Validation runs | fault-free runs 301 to 400, used only to set thresholds                                                          |
| Test runs       | fault-free runs 401 to 500, used only to measure false alarms                                                    |
| Faulty runs     | faults 1 to 20, runs 1 to 20                                                                                     |
| Standardization | subtract each channel's mean and divide by its standard deviation (`ddof=1`), both computed on the training runs |
| Rows to score   | every row of the validation, test and faulty runs                                                                |

### PCA monitor (first week)

1. Standardize the training data. Compute the principal components of its covariance matrix
   (on standardized data this is the correlation matrix).
2. Keep the smallest number of components $k$ whose eigenvalues add up to at least 90 % of the
   total. Report $k$ in the report.
3. For each standardized row $z$, with $P$ the $k$ retained loading vectors and $\lambda_i$ their
   eigenvalues, compute the score $t = P^\top z$ and

   $$T^2 = \sum_{i=1}^{k} \frac{t_i^2}{\lambda_i}, \qquad \text{SPE} = \lVert z - P P^\top z \rVert^2$$

`sklearn.decomposition.PCA` gives the same model: fit it on the standardized training data,
use `explained_variance_` for $\lambda_i$, and `inverse_transform(transform(z))` for $P P^\top z$.
Write `results/scores_pca.parquet` with columns `faultNumber`, `simulationRun`, `sample`, `T2`, `SPE`.
Fault-free rows have `faultNumber` 0.

### Forecast monitor (first week)

1. Inside each run, build a table whose features are the standardized rows at $t-1$ and $t-2$
   (104 columns) and whose targets are the standardized row at $t$ (52 columns). The first two
   samples of each run have no score.
2. Fit one `Ridge(alpha=1.0)` on the training runs, predicting all 52 targets at once.
3. On the training runs, compute each channel's residual standard deviation (`ddof=1`).
4. For every scored row, divide each channel's residual by that standard deviation, square,
   and sum over the 52 channels. That sum is the score.

This is the detector from Lecture 8, applied to every channel at once. Write
`results/scores_ridge.parquet` with columns `faultNumber`, `simulationRun`, `sample`, `score`.

### Evaluation (second week)

1. **Thresholds.** For each of the three statistics (`T2`, `SPE`, `ridge`), the threshold is
   `numpy.quantile(scores, 0.99)` over the validation runs, with NumPy's default interpolation.
   Write `results/thresholds.csv` with columns `detector`, `threshold`.
2. **Alarms.** A sample is in alarm when it and the two samples before it all exceed the
   threshold. Alarms never cross a run boundary.
3. **Metrics**, for each detector:
   - **false-alarm rate:** the share of samples in alarm over the test runs, reported as fault 0;
   - **detection rate:** for each faulty run, the share of samples after sample 20 that are in
     alarm, averaged over the 20 runs;
   - **detection delay:** for each faulty run, minutes from sample 20 to the first alarm after
     it; report the median over the runs that alarmed, and how many runs never alarmed.

Write `results/detection.csv` with columns `fault`, `detector`, `detection_rate`,
`median_delay_min`, `runs_missed`: 21 rows for each detector, faults 0 to 20.

### Diagnosis (second week)

For the SPE and ridge detectors, a statistic that is a sum of squares splits naturally into one
term per channel. That split is called a **contribution**.

- SPE: the contribution of channel $j$ is $(z_j - (P P^\top z)_j)^2$.
- Ridge: the contribution of channel $j$ is its squared standardized residual.

For each fault and each of the two detectors, average the contributions over every sample that
is in alarm after sample 20, across the 20 runs. Write `results/contributions.csv` with columns
`fault`, `detector`, `rank`, `channel`, `contribution`, keeping the top five channels per fault
and detector. Then check a few of them against what the fault physically is, from the fault
list. A contribution plot points at where the fault shows up, which is not always where it
started.

## The report

`REPORT.pdf`, **four pages maximum** plus an appendix, written by the whole team. Write it in
whatever you like (Markdown, a notebook, LaTeX, a word processor), but hand in a PDF so the
figures come with it. The sections, in this order:

1. **The plant and the task.** ~Two paragraphs. This should read such in a way that a reader who has not taken this course before, would be able to clearly follow what is the proposed work here.
2. **The two detectors.** How each works, $k$ for the PCA model, and one plot of each statistic
   on one fault run with its threshold.
3. **Results.** The detection table either as a figure or table: detection rate and median delay per
   fault for all three statistics, and the false-alarm rates.
4. **Comparison.** Which faults one detector catches and the other does not, and why, given what
   each responds to.
5. **Faults nobody catches.** Which faults all three miss (if applicable), and what that says about the data.
   Check your answer against the literature under Resources.
6. **Diagnosis.** For three faults, which channels drive the alarm and whether that matches the
   fault's description.
7. **Limits.** What this setup cannot tell you about a real plant.
8. **AI use.** Disclosing generative-AI use. Be clear and transparent on what you used and why.

**Appendix: contributions.** One entry per member, with their Andrew ID, saying what they
contributed to the project: which parts of the code, which sections of the report, which
analyses. This is where you report your own roles, and it does not count toward the four pages.

## Submit

The team makes two submissions, once each, to two separate Canvas assignments.

1. **The report.** `REPORT.pdf`.
2. **The evidence.** One member puts the team's code and `results/` files in a folder, with the
   data in `data/`, runs the evidence script from that folder with every member's Andrew ID, and
   uploads the `miniproject-evidence.pdf` it writes:

```bash
uv run --no-project https://kitchingroup.cheme.cmu.edu/f26-06763/miniproject-evidence.py \
    --andrew-ids id1 id2 id3 id4 --team "Team name"
```

- The script rebuilds both detectors from the data files and compares them with yours. It does
  not run your code and does not download anything.
- It checks the team's files as a whole and does not read the report.
- **Read the PDF before uploading.** A failing check is a reason to fix it and rerun.
- The PDF prints the script's sha256, which matches
  <https://kitchingroup.cheme.cmu.edu/f26-06763/miniproject-evidence.py.sha256>.

## Grading

The team is scored out of 15, and every member receives the team's score.

| Part                         | Points | Decided by                                                                                                   |
| ---------------------------- | ------ | ------------------------------------------------------------------------------------------------------------ |
| **Detectors**                | 5      | the evidence script: both score files checked against its own rebuild                                        |
| **Evaluation and diagnosis** | 5      | the evidence script: thresholds, the detection table, contributions, and the undetectable faults             |
| **REPORT.pdf**               | 5      | your TA, for the team, with adjustments for an individual's contribution based on the contributions appendix |

Within each scripted part, the points are split evenly over that part's checks.

The checks:

| Part                     | Checks                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Detectors                | `scores_pca.parquet` present; every required row scored and no others; T² and SPE within 0.1 % of the rebuild on 99.9 % of rows; the same for `scores_ridge.parquet` and its score; a ridge fit in the code                                                                                                                                                                                                                            |
| Evaluation and diagnosis | `thresholds.csv` for all three statistics; thresholds equal to the 99th percentile of the team's own validation scores; `detection.csv` equal to a recomputation from the team's own scores and thresholds, with false-alarm rows and faults 1 to 20 for all three statistics; faults 3, 9 and 15 reported as not detected; `contributions.csv` whose top channel per fault and detector matches the rebuild for at least 90 % of them |

The evaluation checks use the team's own scores, so an error in a detector costs points once,
under Detectors, and not again in every table built from it. Projects are group work and the
course's automatic grace days do not apply.

## AI use

Generative AI is allowed with disclosure in the report. Every member must be able to explain
the code they wrote and the team's comparison. Editing a generated PDF by hand is falsifying
a submission.

## A one-page primer on PCA monitoring

PCA finds the directions along which the standardized data vary most. On a plant with 52
channels and a few underlying drivers, most of the variation lies in a much smaller number of
directions, because the channels move together. PCA monitoring asks two questions of every new
sample.

**Is it unusual _within_ the normal pattern?** Project the sample onto the $k$ retained
directions. $T^2$ is the squared distance from the centre in that subspace, with each direction
scaled by its own variance $\lambda_i$, so a large move along a direction that normally barely
moves counts for more. It catches a plant that is still behaving like itself, only more so.

**Is it unusual _outside_ the normal pattern?** The part of the sample the $k$ directions
cannot reproduce is the residual, and SPE (the squared prediction error, also called $Q$) is its
squared length. A large SPE means the channels have stopped moving together the way they
normally do: a relationship between them has broken. MacGregor and Kourti (1995) recommend
monitoring both charts together for this reason.

**Where the threshold comes from.** Textbooks give theoretical limits: an $F$-distribution
limit for $T^2$ and the Jackson and Mudholkar approximation for SPE, both in Russell, Chiang
and Braatz (2000). Those limits assume independent, normally distributed samples, and plant data
sampled every 3 minutes are neither. This project uses the empirical 99th percentile on held-out
normal runs instead, which makes no such assumption and is directly comparable across the three
statistics.

**What PCA cannot see.** A fault that moves the plant in a way that looks exactly like normal
variation, or that the controllers absorb completely, leaves both statistics unchanged. Some of
the twenty faults are like that, and the literature says which.

A minimal sketch, for orientation only; the recipe above is what is graded:

```python
lam, vecs = np.linalg.eigh(np.cov(Z_train, rowvar=False))  # ascending order
lam, vecs = lam[::-1], vecs[:, ::-1]                      # largest first
P, lam_k = vecs[:, :k], lam[:k]
t = z @ P                                                 # scores of one sample
```

## Resources

- Russell, Chiang and Braatz (2000), [Fault detection in industrial processes using canonical variate analysis and dynamic principal component analysis](https://web.mit.edu/braatzgroup/36_Fault_detection_in_industrial_processes_using_canonical_variate_analysis_and_dynamic_principal_component_analysis.pdf), _Chemometrics and Intelligent Laboratory Systems_ 51. The $T^2$ and $Q$ definitions and limits, and a detection study on this plant that singles out three faults (author's copy).
- Chiang, Russell and Braatz (2000), [Fault diagnosis in chemical processes using Fisher discriminant analysis, discriminant partial least squares, and principal component analysis](https://web.mit.edu/braatzgroup/35_Fault_diagnosis_in_chemical_processes_using_Fisher_discriminant_analysis_discriminant_partial_least_squares_and_principal_component_analysis.pdf). Table 1 lists the twenty faults; the paper compares $T^2$ and $Q$ for telling faults apart (author's copy).
- De Ketelaere, Hubert and Schmitt, [A review of PCA-based statistical process monitoring methods for time-dependent, high-dimensional data](https://wis.kuleuven.be/stat/robust/papers/2013/deketelaere-review.pdf). The clearest open derivation of both control limits, plus the dynamic and moving-window variants (authors' preprint).
- Severson, Chaiwatanodom and Braatz (2016), [Perspectives on process monitoring of industrial systems](https://web.mit.edu/braatzgroup/Severson_ARC_2016.pdf), _Annual Reviews in Control_ 42. A readable review of where PCA monitoring sits among the alternatives (author's copy).
- Westerhuis, Gurden and Smilde (2000), [Generalized contribution plots in multivariate statistical process monitoring](https://three-mode.leidenuniv.nl/pdf/w/westerhuis_etal2000cils.pdf), _Chemometrics and Intelligent Laboratory Systems_ 51. What contribution plots show, and the "smearing" that makes them misleading (third-party copy).
- BibMon, [PCA for fault detection in the Tennessee Eastman process](https://bibmon.readthedocs.io/en/latest/tutorial_tep.html). A worked tutorial on this plant with SPE and contribution heatmaps; it uses its own preprocessing, so do not expect identical numbers.
- scikit-learn, [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) and [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).
- Lyu, Botcha, Kulkarni, Pagaria, Alves, Sunshine and Kitchin (2026), [Benchmarking machine learning fault detection methods on the Tennessee Eastman process dataset](https://doi.org/10.26434/chemrxiv.10001628/v1), ChemRxiv. Supervised methods on the same data, for contrast with this project's unsupervised setup.
- Rieth, Amsel, Tran and Cook (2017), [Additional Tennessee Eastman process simulation data](https://doi.org/10.7910/DVN/6C3JR1). The source of both files.
- [Lecture 8](../lectures/l08/notes.md), the section on residuals, for the forecast-residual detector and the false-alarm trade-off.

## Stretch - Extra, not required (not graded)

- Replace the ridge forecaster with a nonlinear ML model of your choice, keep everything else
  fixed, and report whether any fault moves from missed to caught.
- Run the detectors against ten minutes of the live plant stream from Assignment 3, and report
  what the alarms say about its disturbances.
- Add lagged copies of the channels to the PCA model (dynamic PCA, in the Russell et al. paper)
  and compare.
