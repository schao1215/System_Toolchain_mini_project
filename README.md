# System Toolchain Mini Project

Detecting faults in the Tennessee Eastman Process without any labelled fault examples. Two detectors are trained only on normal operation, then scored on twenty faults.

這個專案在沒有故障標籤的情況下，偵測 Tennessee Eastman Process（TEP）的故障。兩個偵測器都只用正常運轉的資料訓練，再拿去對二十種故障打分。

Course handout: [`assignment.md`](assignment.md). A walkthrough of the task and the grading rules is in [`assignment-explained.md`](assignment-explained.md).

作業原文在 [`assignment.md`](assignment.md)。任務內容、要交的檔案和評分邏輯的中文說明在 [`assignment-explained.md`](assignment-explained.md)。

## English

### What this project does

A plant runs normally almost all the time, and the next fault is usually a kind that has never been recorded. A classifier trained on old fault labels cannot see that new kind. This project learns what normal operation looks like, and raises an alarm when a new sample stops looking normal.

Both detectors use the same 52 channels (`xmeas_1`–`xmeas_41` and `xmv_1`–`xmv_11`) and the same split:

| Split | Runs | Used for |
| --- | --- | --- |
| Training | fault-free 1–300 | standardize (`ddof=1`) and fit the models |
| Validation | fault-free 301–400 | thresholds only: `numpy.quantile(scores, 0.99)` |
| Test | fault-free 401–500 | false-alarm rate, reported as fault 0 |
| Faulty | faults 1–20, runs 1–20 | detection rate, delay, contributions |

Samples are 3 minutes apart. A run is 500 samples (25 hours). In a faulty run, samples 1–20 are still normal and the fault starts at sample 21.

A sample is in alarm only when it and the two samples before it all exceed the threshold. The window never crosses a run boundary.

### The two detectors

**PCA.** Standardize the training runs, then keep the smallest number of principal components whose eigenvalues sum to at least 90% of the total. On this data that cutoff is **k = 31** (about 90.2% of the training variance). For a standardized row `z` and loadings `P`:

- `t = P.T @ z`
- `T2 = sum(t_i^2 / lambda_i)` over the retained components. Large when the plant still moves in its usual directions, but by an unusual amount.
- `SPE = ||z - P P.T z||^2`. Large when the channels stop moving together.

**Ridge forecast residual.** Inside each run, predict the standardized row at time `t` from the rows at `t-1` and `t-2` (104 features, 52 targets). One `Ridge(alpha=1.0)` is fit on the training runs. Each residual is divided by that channel's training residual standard deviation (`ddof=1`), squared, and summed. The first two samples of every run have no score, because `t-2` does not exist.

### Results worth knowing before you read the report

Thresholds (validation 99th percentile): T2 **51.80**, SPE **11.77**, ridge **112.4**.

False-alarm rate on the test runs: T2 **0.0005**, SPE **0**, ridge **0**.

Faults **3, 9, and 15** are not detected. SPE and ridge miss all 20 runs of each. T2's detection rate on those three is about 0.0002, the same scale as its own false-alarm rate. Those three are the disturbances the Tennessee Eastman literature calls unobservable: a D-feed temperature step, random variation of that same temperature, and a sticking condenser cooling-water valve. Fault 19 is also weak on all three statistics.

The detectors separate on faults the controllers absorb:

- Fault 4 (reactor cooling-water inlet temperature, step): T2 detection rate 0.091, SPE 0.994, ridge 0.969. The temperature loop pulls the measurements back toward the normal cloud, but the valve–temperature relationship stays broken, so SPE and the forecast residual stay large. The top contribution channels are `xmv_10` and `xmeas_9`.
- Fault 5 (condenser cooling-water inlet temperature, step): just after the fault, all three statistics cross the line. Late in run 1, T2 and SPE fall back under it and the ridge score does not. Over 20 runs the rates are T2 0.332, SPE 0.125, ridge 0.996.

`REPORT.pdf` is the four-page writeup plus a contributions appendix. The figure is fault 7, run 1, where none of the three statistics is absorbed.

### Layout

```text
src/miniproject/
  config.py         shared split, channel names, fault descriptions
  data.py           checksums, load, standardize on training runs
  pca_monitor.py    PCA, T2, SPE
  ridge_monitor.py  lag table and Ridge(alpha=1.0)
  evaluate.py       99th-percentile thresholds, three-sample alarms, detection table
  diagnose.py       top-five channel contributions for SPE and ridge
  plots.py          one fault run of each statistic
  report.py         REPORT.pdf
  pipeline.py       runs the steps above
data/               the two parquet files and SHA256SUMS
results/            score files and tables
figures/            the fault-run figure embedded in the report
```

### How to run

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. From this directory:

```bash
uv sync
uv run python -m miniproject
```

`uv sync` creates `.venv` and installs the packages in `pyproject.toml` (`numpy`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `fpdf2`). The entry point does not download data and does not call the course evidence script.

The pipeline checks `data/SHA256SUMS` before it fits anything. If you need to download the files again:

```bash
mkdir -p data && cd data
for f in tep_fault_free_training.parquet tep_faulty_training_runs01-20.parquet SHA256SUMS; do
  curl -fLO "https://kitchin-services.cheme.cmu.edu/f26-06763/data/$f"
done
shasum -a 256 -c SHA256SUMS
```

### Files the pipeline writes

| Path | Contents |
| --- | --- |
| `results/scores_pca.parquet` | `faultNumber`, `simulationRun`, `sample`, `T2`, `SPE`. Fault-free rows use `faultNumber` 0. 300,000 rows (validation, test, and every faulty row). |
| `results/scores_ridge.parquet` | `faultNumber`, `simulationRun`, `sample`, `score`. 298,800 rows. Samples 1 and 2 of each run are absent. |
| `results/thresholds.csv` | `detector`, `threshold` for `T2`, `SPE`, and `ridge`. |
| `results/detection.csv` | 21 rows (faults 0–20) for each detector: `detection_rate`, `median_delay_min`, `runs_missed`. Fault 0 is the false-alarm rate. Delay is minutes from sample 20 to the first later alarm, median over runs that alarmed. |
| `results/contributions.csv` | Top five channels per fault for SPE and ridge, averaged over alarming samples after sample 20. Faults with no such samples are omitted. |
| `figures/fault7_run1_statistics.png` | T2, SPE, and the ridge score on fault 7, run 1, with the threshold. |
| `REPORT.pdf` | Sections required by the handout, at most four pages, plus the contributions appendix. |

### What is left before submission

The course asks for two Canvas uploads: `REPORT.pdf`, and `miniproject-evidence.pdf` produced by the course evidence script. This repository does not run that script. Before handing in the report, replace `ANDREW_ID` in the appendix and add an entry for each teammate.

---

## 中文

### 這個專案在做什麼

工廠幾乎一直正常運轉，下一次故障又常常是以前沒記錄過的型態。用舊的故障標籤訓練分類器，看不到這種新故障。所以這裡反過來：只用正常資料學「正常長什麼樣子」，新的一列不再像正常時就拉警報。

兩個偵測器共用 52 個通道（`xmeas_1`–`xmeas_41` 與 `xmv_1`–`xmv_11`），也共用同一套切分：

| 切分 | Run | 用途 |
| --- | --- | --- |
| 訓練 | 無故障 1–300 | 標準化（`ddof=1`）並擬合模型 |
| 驗證 | 無故障 301–400 | 只用來定門檻：`numpy.quantile(scores, 0.99)` |
| 測試 | 無故障 401–500 | 誤報率，在表裡記成 fault 0 |
| 故障 | 故障 1–20，每個 run 1–20 | 偵測率、延遲、通道貢獻 |

取樣間隔 3 分鐘。一個 run 有 500 個 sample，也就是 25 小時。故障 run 裡，sample 1–20 仍是正常，故障從 sample 21 開始。

某個 sample 要進入警報，必須它自己和前面兩個 sample 都超過門檻。這個視窗不能跨過 run 的邊界。

### 兩個偵測器

**PCA。** 先用訓練 run 做標準化，再保留最少的主成分，使特徵值加總至少佔總變異的 90%。這份資料上的結果是 **k = 31**（約 90.2%）。對標準化後的一列 `z`、負荷矩陣 `P`：

- `t = P.T @ z`
- `T2` 是保留成分上 `t_i^2 / lambda_i` 的和。工廠仍沿著平常的方向在動，只是幅度異常時，T2 會變大。
- `SPE = ||z - P P.T z||^2`。通道不再照平常的方式一起動、保留成分重建不回來時，SPE 會變大。

**Ridge 預測殘差。** 在每個 run 裡面，用 `t-1` 和 `t-2` 的標準化列（104 欄）預測 `t` 的標準化列（52 欄）。訓練 run 上只擬合一個 `Ridge(alpha=1.0)`。每個通道的殘差先除以該通道在訓練集上的殘差標準差（`ddof=1`），再平方、對 52 個通道加總。每個 run 的前兩個 sample 沒有分數，因為 `t-2` 不存在。

### 看報告之前可以先知道的結果

驗證集第 99 百分位的門檻：T2 **51.80**、SPE **11.77**、ridge **112.4**。

測試 run 的誤報率：T2 **0.0005**，SPE **0**，ridge **0**。

故障 **3、9、15** 沒有被偵測到。SPE 和 ridge 對這三個故障的 20 個 run 全部沒有警報。T2 在這三個故障上的偵測率約 0.0002，和它自己的誤報率同一量級。文獻把這三個稱為不可觀測：D 進料溫度的階躍、同一溫度的隨機變動，以及冷凝器冷卻水閥卡住。故障 19 在三個統計量上也很弱。

控制器把擾動吸收掉時，兩個偵測器會分開：

- 故障 4（反應器冷卻水入口溫度，階躍）：T2 偵測率 0.091，SPE 0.994，ridge 0.969。溫度迴路把測量拉回正常雲，但閥和溫度的關係仍然是壞的，所以 SPE 和預測殘差一直很大。貢獻最高的通道是 `xmv_10` 和 `xmeas_9`。
- 故障 5（冷凝器冷卻水入口溫度，階躍）：剛故障時三個統計量都超過門檻。run 1 的後段，T2 和 SPE 回到門檻以下，ridge 沒有。20 個 run 的偵測率是 T2 0.332、SPE 0.125、ridge 0.996。

`REPORT.pdf` 是作業要求的四頁報告，另加貢獻附錄。圖用的是故障 7 的 run 1，三個統計量都沒有被控制器吸收。

### 目錄

```text
src/miniproject/
  config.py         共用的資料切分、通道名稱、故障說明
  data.py           checksum、讀檔、用訓練 run 做標準化
  pca_monitor.py    PCA、T2、SPE
  ridge_monitor.py  落後項表格與 Ridge(alpha=1.0)
  evaluate.py       第 99 百分位門檻、連續三點警報、偵測表
  diagnose.py       SPE 與 ridge 的前五個通道貢獻
  plots.py          一條故障 run 上的三個統計量
  report.py         REPORT.pdf
  pipeline.py       依序執行上面的步驟
data/               兩個 parquet 與 SHA256SUMS
results/            分數檔與表格
figures/            嵌在報告裡的故障 run 圖
```

### 怎麼跑

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。在這個目錄：

```bash
uv sync
uv run python -m miniproject
```

`uv sync` 會建立 `.venv`，並安裝 `pyproject.toml` 裡的套件（`numpy`、`pandas`、`pyarrow`、`scikit-learn`、`matplotlib`、`fpdf2`）。這個進入點不會下載資料，也不會呼叫課程的 evidence 腳本。

擬合之前會先檢查 `data/SHA256SUMS`。若要重新下載：

```bash
mkdir -p data && cd data
for f in tep_fault_free_training.parquet tep_faulty_training_runs01-20.parquet SHA256SUMS; do
  curl -fLO "https://kitchin-services.cheme.cmu.edu/f26-06763/data/$f"
done
shasum -a 256 -c SHA256SUMS
```

### 管線寫出的檔案

| 路徑 | 內容 |
| --- | --- |
| `results/scores_pca.parquet` | `faultNumber`、`simulationRun`、`sample`、`T2`、`SPE`。無故障列的 `faultNumber` 是 0。共 300,000 列（驗證、測試，以及每一列故障資料）。 |
| `results/scores_ridge.parquet` | `faultNumber`、`simulationRun`、`sample`、`score`。共 298,800 列。每個 run 的 sample 1 和 2 不在檔裡。 |
| `results/thresholds.csv` | `T2`、`SPE`、`ridge` 的 `detector` 與 `threshold`。 |
| `results/detection.csv` | 每個偵測器 21 列（fault 0–20）：`detection_rate`、`median_delay_min`、`runs_missed`。Fault 0 是誤報率。延遲是從 sample 20 到之後第一個警報的分鐘數，只對有警報的 run 取中位數。 |
| `results/contributions.csv` | SPE 與 ridge 在每個故障上貢獻最高的五個通道，平均的是 sample 20 之後且正在警報的列。完全沒有這種列的故障不會出現。 |
| `figures/fault7_run1_statistics.png` | 故障 7、run 1 上的 T2、SPE 和 ridge 分數，以及門檻。 |
| `REPORT.pdf` | 作業指定的章節，正文最多四頁，另加貢獻附錄。 |

### 繳交前還要做的事

課程要交兩份東西到 Canvas：`REPORT.pdf`，以及課程 evidence 腳本產生的 `miniproject-evidence.pdf`。這個倉庫沒有執行那個腳本。交報告之前，把附錄裡的 `ANDREW_ID` 換成自己的 Andrew ID，並為每一位組員各寫一則貢獻說明。
