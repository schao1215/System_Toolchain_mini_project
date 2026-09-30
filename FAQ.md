# FAQ

Short answers to the design questions behind this repository. The formulas are the ones in `src/miniproject/`. The handout is [`assignment.md`](assignment.md).

這個檔案回答這個倉庫背後的設計問題。公式和 `src/miniproject/` 裡的實作一致。作業原文是 [`assignment.md`](assignment.md)。

## English

### 1. What does the data look like, how much is there, and what is the goal?

Two Parquet files, already in `data/`. Each row is one sample of the simulated plant.

| File | Rows | On disk | What a row is |
| --- | --- | --- | --- |
| `tep_fault_free_training.parquet` | 250,000 | about 24 MB | 500 normal runs, 500 samples each |
| `tep_faulty_training_runs01-20.parquet` | 200,000 | about 20 MB | 20 faults, 20 runs each, 500 samples each |

Every row has 55 columns: `faultNumber`, `simulationRun`, `sample`, then 52 process channels (`xmeas_1`–`xmeas_41` and `xmv_1`–`xmv_11`). The 41 `xmeas` columns are measured flows, levels, pressures, temperatures, and compositions. The 11 `xmv` columns are valve positions. Samples are 3 minutes apart, so one run is 25 hours.

In the faulty file the disturbance is injected one hour in. Samples 1–20 of a faulty run are still normal. Samples 21–500 are the fault.

The goal is not to classify a fault from labelled examples. Faults are rare, and the next one is usually a kind that has never been recorded, so a classifier trained on old labels cannot see it. Both detectors are fit on fault-free runs 1–300 only. They learn what normal looks like. A later row raises an alarm when it stops looking normal. Runs 301–400 set the threshold. Runs 401–500 measure false alarms. The twenty faults are used only for scoring.

### 2. Why PCA, and how is k chosen?

The 52 channels do not move independently. A few underlying drivers (feeds, recycle, reactor temperature, the controllers) push many sensors at once, so most of the variation lies in far fewer than 52 directions. PCA finds those directions on the standardized training runs. Monitoring then asks two questions of every new row: is it unusual *inside* that normal pattern, and is it unusual *outside* it? That is $T^{2}$ and SPE below. Without the reduction, a single threshold on 52 raw channels does not know which moves are the usual ones.

$k$ is not tuned to the faults. It is the smallest number of components whose eigenvalues add up to at least 90% of the total:

$$
k = \min\left\lbrace m : \frac{\sum_{i=1}^{m} \lambda_i}{\sum_{i=1}^{52} \lambda_i} \ge 0.90 \right\rbrace.
$$

In code this is the cumulative sum of `explained_variance_ratio_` after `PCA(svd_solver="full")` on the standardized training matrix. The search keeps a component that lands exactly on 90%. On this data, $k = 31$, and those 31 components carry about 90.2% of the training variance. The other 21 directions are the residual that SPE watches.

### 3. What are T2, SPE, and Ridge, and how were they designed?

All three scores are computed on rows standardized with the training mean and the training standard deviation only (`ddof=1`, runs 1–300):

$$
z_j = \frac{x_j - \mu_j}{\sigma_j}.
$$

Validation, test, and faulty rows use that same $\mu$ and $\sigma$. They are not re-centered on their own data.

**$T^{2}$** asks whether the point is unusual inside the normal pattern. Project the standardized row onto the $k$ retained loadings $P$ and scale each coordinate by the variance of that direction:

$$
t = P^{\top} z, \qquad
T^{2} = \sum_{i=1}^{k} \frac{t_i^{2}}{\lambda_i}.
$$

A large move along a direction that normally barely moves counts more, because its $\lambda_i$ is small. $T^{2}$ catches a plant that is still behaving like itself, only more so. $\lambda_i$ is `explained_variance_` from scikit-learn, which is the eigenvalue of the covariance with divisor $n-1$.

**SPE** (squared prediction error, also called $Q$) asks whether the point is unusual outside that pattern:

$$
\mathrm{SPE} = \lVert z - P P^{\top} z \rVert^{2}.
$$

It is the squared length of whatever the $k$ components cannot reconstruct. A large SPE means the channels have stopped moving together the way they normally do: a relationship between them has broken. The code projects $z$ directly, $t = z P_{k}$, rather than letting PCA subtract a second mean. The training rows already have mean zero, so the two versions match, and the formula stays the one in the handout.

**Ridge** is a different question: can the recent past predict the present? Inside each run the features are the standardized rows at $t-1$ and $t-2$ (104 columns, $t-1$ first) and the target is the standardized row at $t$ (52 columns). One `Ridge(alpha=1.0)` is fit on the training runs and predicts all 52 channels at once. Sklearn's default intercept is left on; `alpha=1.0` is the only setting the handout names. On the training runs, each channel gets a residual standard deviation $s_j$ with `ddof=1`. The score of a later row is

$$
\mathrm{score}_t = \sum_{j=1}^{52} \left( \frac{z_{t,j} - \hat{z}_{t,j}}{s_j} \right)^{2}.
$$

Dividing by $s_j$ stops a channel that is hard to forecast even in normal operation from dominating the sum. Samples 1 and 2 of a run have no score, because $t-2$ does not exist, and a lag never crosses into another run. This is the Lecture 8 residual detector, applied to every channel at once. It responds to a surprise in the dynamics. $T^{2}$ and SPE respond to where the operating point sits.

The threshold for each score is `numpy.quantile(validation_scores, 0.99)` on runs 301–400, with NumPy's default interpolation. A sample alarms only when it and the two samples before it all exceed that line.

### 4. What is the logic of a contribution?

$T^{2}$ mixes the channels inside the subspace, so it does not split into one term per sensor. SPE and the ridge score are both sums of squares, so each term already belongs to one channel. That term is the contribution.

$$
c_j^{\mathrm{SPE}} = \left( z_j - (P P^{\top} z)_j \right)^{2}, \qquad
c_j^{\mathrm{ridge}} = \left( \frac{z_{t,j} - \hat{z}_{t,j}}{s_j} \right)^{2}.
$$

SPE's term is the squared reconstruction error of channel $j$. Ridge's term is that channel's squared standardized residual. The sum of the 52 contributions is the score itself.

For each fault and each of those two detectors, the contributions are averaged over every sample that is in alarm after sample 20, across the 20 runs. Samples that are over the line but not yet an alarm (the three-in-a-row rule has not fired) are left out, and so are the 20 normal samples at the start of the run. The five largest average contributions are written to `results/contributions.csv`. If a fault never alarms after sample 20, it has no row: faults 3, 9, and 15 are in that group for both SPE and ridge.

The ranking says where the fault shows up, which is not always where it started. A feed fault can move the reactor and the recycle, and a channel correlated with the disturbed one inherits part of the score (the smearing in Westerhuis, Gurden and Smilde, 2000). Fault 6 is the example in the report: the lost stream is the A feed, but SPE's largest term is the compressor recycle valve.

### 5. How does a row become an alarm, and what do the detection numbers mean?

Exceeding the threshold once is not an alarm. The row and the two samples before it, in the same run, must all be strictly above the threshold. The test-set false-alarm rate is the share of those alarmed samples on fault-free runs 401–500, stored as fault 0.

For a faulty run, the detection rate is the share of samples after sample 20 that are in alarm. The table reports the average of that share over the 20 runs. The delay is the minutes from sample 20 to the first alarm after it, $(s - 20) \times 3$, and the table reports the median over the runs that did alarm, plus how many runs never alarmed.

### 6. Why do faults 3, 9, and 15 produce no alarm?

Fault 3 is a step in D-feed temperature, fault 9 is random variation of that same temperature, and fault 15 is a sticking condenser cooling-water valve. On the 52 recorded channels those runs look like normal operation: the temperature disturbance barely appears in the measurements, and the level and temperature loops can absorb the sticking valve. SPE and ridge miss all 20 runs. $T^{2}$ alarms on one run of each, at a rate of about 0.0002, which is the same scale as its false-alarm rate on the normal test runs. The literature calls these three unobservable. The model is not failing to implement the recipe; the recorded data do not contain a departure from the normal pattern.

---

## 中文

### 1. 資料長什麼樣子、有多少、檔案多大、目標是什麼？

資料是 `data/` 裡的兩個 Parquet。每一列是模擬工廠的一個 sample。

| 檔案 | 列數 | 磁碟大小 | 一列是什麼 |
| --- | --- | --- | --- |
| `tep_fault_free_training.parquet` | 250,000 | 約 24 MB | 500 個正常 run，每個 500 個 sample |
| `tep_faulty_training_runs01-20.parquet` | 200,000 | 約 20 MB | 20 個故障、每個 20 個 run、每個 run 500 個 sample |

每一列 55 欄：`faultNumber`、`simulationRun`、`sample`，再加上 52 個製程通道（`xmeas_1`–`xmeas_41` 與 `xmv_1`–`xmv_11`）。41 個 `xmeas` 是流量、液位、壓力、溫度與組成的量測。11 個 `xmv` 是閥位。取樣間隔 3 分鐘，所以一個 run 是 25 小時。

故障檔裡，擾動在一小時後才注入。故障 run 的 sample 1–20 仍是正常，sample 21–500 才是故障。

目標不是拿標好的故障去分類。故障很少，下一次又通常是沒記錄過的型態，用舊標籤訓練的分類器看不到它。兩個偵測器都只在無故障 run 1–300 上擬合，學的是正常長什麼樣子。後面的列不再像正常時就拉警報。Run 301–400 只拿來定門檻。Run 401–500 只拿來量誤報。二十種故障只用於打分，不參與訓練。

### 2. 為什麼要做 PCA？k 怎麼訂？

這 52 個通道不是各自亂走。進料、循環、反應器溫度、控制器這少數幾個驅動力會同時推動很多感測器，所以大部分變異落在遠少於 52 個方向上。PCA 在標準化後的訓練 run 上找出這些方向。監控再對每一列新資料問兩件事：它在正常模式**裡面**是否異常，以及在正常模式**外面**是否異常。這就是下面的 $T^{2}$ 和 SPE。若不做這個降維，對 52 個原始通道各設一個門檻，並不知道哪些移動本來就是正常的。

$k$ 不是對著故障調出來的。它是特徵值加總至少佔總變異 90% 的最少成分數：

$$
k = \min\left\lbrace m : \frac{\sum_{i=1}^{m} \lambda_i}{\sum_{i=1}^{52} \lambda_i} \ge 0.90 \right\rbrace.
$$

程式裡是對標準化訓練矩陣做 `PCA(svd_solver="full")`，再對 `explained_variance_ratio_` 做累加。剛好落在 90% 的那個成分也會被留下。這份資料上 $k = 31$，這 31 個成分大約帶著訓練變異的 90.2%。剩下的 21 個方向就是 SPE 在看的殘差。

### 3. T2、SPE、Ridge 分別是什麼？怎麼設計的？

三個分數都打在同一套標準化之後的列上。平均 $\mu$ 和標準差 $\sigma$ 只用訓練 run（1–300），標準差的除數是 $n-1$：

$$
z_j = \frac{x_j - \mu_j}{\sigma_j}.
$$

驗證、測試、故障列都用這同一組 $\mu$ 和 $\sigma$，不會用自己的資料重新對中。

**$T^{2}$** 問的是：這一點在正常模式裡面是否異常。把標準化列投影到保留的 $k$ 個負荷 $P$ 上，再除以該方向自己的變異：

$$
t = P^{\top} z, \qquad
T^{2} = \sum_{i=1}^{k} \frac{t_i^{2}}{\lambda_i}.
$$

平常幾乎不動的方向，只要動了一點，因為 $\lambda_i$ 很小，分數會被放大。$T^{2}$ 抓的是「工廠還是自己，只是做得太過」。$\lambda_i$ 用的是 scikit-learn 的 `explained_variance_`，也就是共變異數矩陣、除數為 $n-1$ 的特徵值。

**SPE**（平方預測誤差，也叫 $Q$）問的是：這一點在那個模式外面是否異常。

$$
\mathrm{SPE} = \lVert z - P P^{\top} z \rVert^{2}.
$$

它是 $k$ 個成分重建不回來的那一段的平方長度。SPE 大，表示通道不再照平常的方式一起動，通道之間的某個關係壞了。程式直接做 $t = z P_k$，沒有讓 PCA 再減一次平均。訓練列的平均本來就是 0，兩種算法一致，公式也和作業規定的相同。

**Ridge** 問的是另一件事：最近的過去能不能預測現在。在每個 run 裡面，特徵是 $t-1$ 與 $t-2$ 的標準化列（104 欄，$t-1$ 在前），目標是 $t$ 的標準化列（52 欄）。訓練 run 上只擬合一個 `Ridge(alpha=1.0)`，一次預測全部 52 個通道。截距留著 sklearn 的預設；作業點名的設定只有 `alpha=1.0`。訓練殘差再為每個通道算一個標準差 $s_j$（`ddof=1`）。之後每一列的分數是

$$
\mathrm{score}_t = \sum_{j=1}^{52} \left( \frac{z_{t,j} - \hat{z}_{t,j}}{s_j} \right)^{2}.
$$

除以 $s_j$ 是為了不讓「平常就很難預測」的通道主導這個和。每個 run 的 sample 1 和 2 沒有分數，因為 $t-2$ 不存在，落後項也不會跨到另一個 run。這就是 Lecture 8 的殘差偵測器，一次用在全部通道上。它回應的是動態上的意外。$T^{2}$ 和 SPE 回應的是操作點落在哪裡。

每個分數的門檻都是 run 301–400 上的 `numpy.quantile(scores, 0.99)`，插值用 NumPy 的預設。一個 sample 要成為警報，必須它自己和前面兩個 sample 都嚴格超過這條線。

### 4. Contribution 的邏輯是什麼？

$T^{2}$ 把通道混在子空間裡，拆不回一個感測器一項。SPE 和 ridge 分數都是平方和，每一項本來就屬於一個通道。那一項就是 contribution。

$$
c_j^{\mathrm{SPE}} = \left( z_j - (P P^{\top} z)_j \right)^{2}, \qquad
c_j^{\mathrm{ridge}} = \left( \frac{z_{t,j} - \hat{z}_{t,j}}{s_j} \right)^{2}.
$$

SPE 的那一項是通道 $j$ 的重建誤差平方。Ridge 的那一項是該通道標準化殘差的平方。52 項加起來就是分數本身。

對每個故障、這兩個偵測器，把 sample 20 之後**而且正在警報**的列，跨 20 個 run 做平均。只是超過門檻、但連續三點規則還沒成立的列不算，run 開頭那 20 個正常 sample 也不算。平均之後最大的五個通道寫進 `results/contributions.csv`。若某個故障在 sample 20 之後從未警報，就沒有這一列：故障 3、9、15 在 SPE 和 ridge 上都是這種情況。

這個排序指出的是故障**表現出來的地方**，不一定是故障**開始的地方**。進料故障會帶動反應器和循環；和被擾動通道相關的感測器也會分到一部分分數（Westerhuis、Gurden 與 Smilde，2000，說的 smearing）。報告裡的故障 6 就是例子：失去的是 A 進料，但 SPE 最大的一項是壓縮機循環閥。

### 5. 一列怎麼變成警報？偵測率與延遲怎麼算？

單次超過門檻不是警報。必須這一列和同一個 run 裡它前面的兩個 sample 都嚴格高於門檻。測試集誤報率是無故障 run 401–500 裡，這種警報列所佔的比例，記成 fault 0。

對一個故障 run，偵測率是 sample 20 之後處於警報的比例。表上的數字是 20 個 run 的平均。延遲是從 sample 20 到之後第一個警報的分鐘數，$(s - 20) \times 3$。表上報的是有警報的那些 run 的中位數，另外記下從未警報的 run 數。

### 6. 為什麼故障 3、9、15 沒有警報？

故障 3 是 D 進料溫度的階躍，故障 9 是同一溫度的隨機變動，故障 15 是冷凝器冷卻水閥卡住。在記錄下來的 52 個通道上，這些 run 看起來就像正常運轉：溫度擾動幾乎沒有出現在量測裡，液位與溫度迴路也可以把卡住的閥吸收掉。SPE 和 ridge 的 20 個 run 全部沒有警報。$T^{2}$ 在每一個故障上只對一個 run 響過，偵測率約 0.0002，和它在正常測試 run 上的誤報率同一量級。文獻把這三個稱為不可觀測。不是公式沒有照作業做，而是這三份記錄裡沒有離開正常模式的訊號。
