# Spec 01: Multi-Head Loss Calibration, Power-Clip Debiasing, and Gradient Variance Stabilization

**Status:** Implemented  
**Target Codebase:** `ChordFormer-Artificial-Dataset-Benchmarking` (`confor_head16.py`, `finetune_real_fold.py`)  
**Cross-Reference:** `probabilistic-composition-generator` [Spec 19 - Empirical Benchmark Mitigations and Distributional Alignment](../../probabilistic-composition-generator/specs/19-empirical-benchmark-mitigations-and-distributional-alignment.md)  
**Audience:** ACR model architecture, multi-task loss modeling, fine-tuning schedules, and benchmark evaluation  

---

## 1. Problem Statement & Empirical Background

Following synthetic pretraining and curriculum replay fine-tuning on the 1,217-song benchmark (McGill Billboard, Isophonics Beatles/Queen, Robbie Williams, Usun, RWC), ChordFormer demonstrated massive recall improvements on underrepresented harmonic extensions (e.g. augmented triads $+17.4\%$, `sus2` $+12.6\%$, altered 9ths and 11ths $+3.7\%\text{--}4.7\%$, `b7` $+7.5\%$).

However, the complete 11,436,357-frame confusion matrix evaluation revealed that the existing `ReweightedLoss(power=0.5, max_clip=10.0)` setup caused severe false-positive flooding across majority classes:

1. **`b7_clamp` Pathology in Head 2 (7th Extension)**:
   - Clamping `b7` weight floor to $\max(0.8 \times w_7, 3.5) = 3.602$ artificially penalized `b7` false negatives 3.6x more than unextended triads (`N`, weight $1.0$).
   - **Empirical Impact**: **1,556,188 frames of clean unextended triads (`N`) (19.52%)** were falsely classified as `b7`.
   - **29.17% of `bb7` (dim7)** and **11.44% of `7` (maj7)** were also misclassified as `b7`, reducing Head 2 frame accuracy by $-2.22\%$.

2. **Extreme Class Weight Disproportion in Head 0 (Triad Quality)**:
   - `sus4` ($w = 5.521$) caused **247,892 frames of `maj`** to be falsely predicted as `sus4` (precision dropped to $28.0\%$).
   - `5` (power chords, $w = 4.976$) caused **168,568 frames of `maj`** to be falsely predicted as `5` (precision dropped to $25.6\%$).
   - `dim` and `aug` hit the $10.0\times$ ceiling, producing over **114,000 false positives** from `maj` and `min`.

3. **Extension Head False-Positive Flooding (Heads 3, 4, 5)**:
   - **Head 4 (Natural 11)**: With weight $8.763$, **202,022 frames of `N`** were falsely tagged as `11`. Out of 232,405 total predictions, **87.3% were false positives** (precision was $12.7\%$).
   - **Head 3 (Altered #9)**: With weight $10.0$, **94.5% of all `#9` predictions were false positives** (33,045 false frames from `N`).
   - **Head 2 (`bb7`)**: With weight $10.0$, precision was only $18.8\%$ (193,263 false frames from `N`).

4. **Multi-Head Task Gradient Imbalance**:
   - In `confor_head16.py`, all 6 task losses were summed with equal $1.0\times$ weights:
     $$\mathcal{L} = \mathcal{L}_{\text{triad}} + \mathcal{L}_{\text{bass}} + \mathcal{L}_{7\text{th}} + \mathcal{L}_{9\text{th}} + \mathcal{L}_{11\text{th}} + \mathcal{L}_{13\text{th}}$$
   - Heads 4 and 5 (11th and 13th extensions) have $>99\%$ `N` frames. The high-penalty gradient spikes from their rare classes injected high variance into the shared Conformer encoder, destabilizing the representations for the foundational harmonic heads (Head 0 Root/Triad and Head 1 Bass).

---

## 2. Mathematical Formulation of Calibrated Loss

To resolve both intra-head class distortion and inter-head task gradient variance, the loss function is calibrated along two complementary axes.

### 2.1 Intra-Head Class Reweighting Calibration
The class loss weight $w_c$ for class $c$ in head $h$ is formulated as:
$$w_c = \min\left(\left(\frac{\max_k N_k}{\max(N_c, 1.0)}\right)^\alpha, C_{\max}\right)$$

Where:
- **Power Exponent ($\alpha$)**: Reduced from $\mathbf{0.50} \rightarrow \mathbf{0.25}$ (fourth root). When class frequency ratios range from $1,000 : 1$ to $10,000 : 1$, $\alpha = 0.50$ produces weights of $31\times$ to $100\times$, forcing reliance on arbitrary clipping. In contrast, $\alpha = 0.25$ smoothly scales rare classes to $3\times\text{--}5\times$ without extreme gradient distortion.
- **Maximum Clip Ceiling ($C_{\max}$)**: Reduced from $\mathbf{10.0} \rightarrow \mathbf{4.0}$. No rare class is permitted to penalize false negatives more than $4\times$ relative to majority classes. This strictly limits false-positive bleeding into `N` and `maj`.
- **`b7_clamp`**: Disabled by default (`b7_clamp = False`). `b7` naturally represents $>20\%$ of real dataset frames and receives its natural calibrated weight ($w_{b7} \approx 1.358$).

#### Weight Comparison: Uncalibrated vs. Calibrated

| Head | Class | Uncalibrated Weight ($\alpha=0.5, C_{\max}=10.0$) | Calibrated Weight ($\alpha=0.25, C_{\max}=4.0$) | Reduction Factor | Expected Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Head 0** | `maj` | 1.000 | 1.000 | $1.0\times$ | Majority anchor |
| **Head 0** | `min` | 1.678 | 1.295 | $0.77\times$ | Balanced triad boundary |
| **Head 0** | `sus4` | **5.521** | **2.350** | **$0.43\times$** | **Stops 247k false positives from `maj`** |
| **Head 0** | `5` (power) | **4.976** | **2.231** | **$0.45\times$** | **Stops 168k false positives from `maj`** |
| **Head 0** | `sus2` | **10.000** | **3.840** | **$0.38\times$** | Stops over-prediction |
| **Head 0** | `dim` | **10.000** | **3.397** | **$0.34\times$** | **Stops 80k false positives from `maj`/`min`** |
| **Head 0** | `aug` | **10.000** | **4.000** | **$0.40\times$** | Balanced boundary |
| **Head 2** | `7` (maj7) | 4.502 | 2.122 | $0.47\times$ | Prevents `maj7` over-sensitivity |
| **Head 2** | `b7` | **3.602** (clamped) | **1.358** | **$0.38\times$** | **Stops 1.55M false positives from `N`** |
| **Head 2** | `bb7` | **10.000** | **4.000** | **$0.40\times$** | **Stops 193k false positives from `N`** |
| **Head 3** | `9` | **4.302** | **2.074** | **$0.48\times$** | **Stops 604k false positives from `N`** |
| **Head 3** | `#9` | **10.000** | **3.859** | **$0.39\times$** | Resolves 94.5% false discovery rate |
| **Head 4** | Natural `11` | **8.763** | **2.960** | **$0.34\times$** | **Stops 202k false positives from `N`** |
| **Head 4** | `#11` | **10.000** | **4.000** | **$0.40\times$** | Maintains high sensitivity with bounded FP |
| **Head 5** | Natural `13` | **6.735** | **2.595** | **$0.39\times$** | Stops false alarms on unextended chords |
| **Head 5** | `b13` | **10.000** | **4.000** | **$0.40\times$** | Bounded alteration penalty |

---

### 2.2 Inter-Head Multi-Task Loss Weighting
The total multi-task loss is weighted across the 6 classification heads using an empirically tuned importance vector $\boldsymbol{\lambda} = [\lambda_0, \lambda_1, \lambda_2, \lambda_3, \lambda_4, \lambda_5]$:

$$\mathcal{L}_{\text{total}} = \sum_{h=0}^5 \lambda_h \cdot \mathcal{L}_h(\hat{\mathbf{y}}_h, \mathbf{y}_h; \mathbf{w}_h)$$

Where:
- $\lambda_0 = \mathbf{1.00}$ (Head 0: Root/Triad — primary harmonic backbone)
- $\lambda_1 = \mathbf{1.25}$ (Head 1: Bass Pitch Class — boosted to overcome the $-1.61\%$ low-end deficit)
- $\lambda_2 = \mathbf{0.80}$ (Head 2: 7th Extension — tempered to prevent dominant 7th gradient surge)
- $\lambda_3 = \mathbf{0.50}$ (Head 3: 9th Extension — scaled to match harmonic presence)
- $\lambda_4 = \mathbf{0.30}$ (Head 4: 11th Extension — scaled down to suppress gradient noise from $>98.7\%$ `N` frames)
- $\lambda_5 = \mathbf{0.20}$ (Head 5: 13th Extension — scaled down to suppress gradient noise from $>99.5\%$ `N` frames)

---

### 2.3 Head 1 (Bass) 13-Class Per-Pitch Balancing
Previously, `counter[1]` collapsed all non-N pitch classes into a single aggregate count, forcing all 12 bass notes (`C` through `B`) to share an identical $1.000$ weight despite significant key distribution differences in real music (guitar keys `A`, `D`, `E`, `G` have $\sim 1.4\text{M}$ frames vs. flat keys `C#`, `Eb`, `F#`, `Ab` with $\sim 550\text{k}$ frames).

When `counter[1]` provides exact 13-class counts (or when computed from fold training frames), `ReweightedLoss` assigns native fourth-root inverse frequency weights per pitch class:
- $w_{\text{Bass}}(A) = 1.000$
- $w_{\text{Bass}}(C\#) = (887,761 / 303,957)^{0.25} = 1.307$
- $w_{\text{Bass}}(N) = 1.296$

This provides a subtle, natural key balance across all 12 tonalities without introducing distortion.

---

## 3. Implementation Details

### 3.1 `confor_head16.py` Changes
1. **`ReweightedLoss.__init__`**:
   - Signature updated with defaults: `power=0.25`, `max_clip=4.0`, `b7_clamp=False`, `head_weights=[1.0, 1.25, 0.8, 0.5, 0.3, 0.2]`.
   - Handles exact 13-class bass weighting if `len(counter[1]) == 13`, with backwards-compatible fallback if `len(counter[1]) == 2`.
2. **`ReweightedLoss.forward`**:
   - Multiplies each conditional cross-entropy head loss by its corresponding $\lambda_h \in \text{head\_weights}$.
3. **`ChordNet.__init__`**:
   - Forwards `power`, `max_clip`, `b7_clamp`, and `head_weights` directly to `ReweightedLoss`.
4. **CLI Parser**:
   - `--power` default updated to `0.25`.
   - `--max_clip` default updated to `4.0`.
   - `--b7_clamp` converted to opt-in legacy flag (`default=False`).
   - `--head_weights` added (default: `'1.0,1.25,0.8,0.5,0.3,0.2'`).

### 3.2 `finetune_real_fold.py` Changes
1. **`finetune_fold` signature & defaults**:
   - Updated defaults: `loss_power=0.25`, `loss_max_clip=4.0`, `loss_b7_clamp=False`, `head_weights=None`.
2. **CLI Parser**:
   - Added `--head_weights` argument with default `'1.0,1.25,0.8,0.5,0.3,0.2'`.
   - Updated `--loss_power` default to `0.25` and `--loss_max_clip` default to `4.0`.
   - Replaced `--no_b7_clamp` with opt-in `--b7_clamp` flag.

### 3.3 `pretrain_synthetic.py` Changes
1. **`pretrain_synthetic` signature**:
   - Added `head_weights=None` parameter and forwarded `head_weights=head_weights` to `ChordNet`.
2. **CLI Parser**:
   - Added `--head_weights` argument (default `'1.0,1.25,0.8,0.5,0.3,0.2'`), ensuring Stage 1 pretraining applies the exact same multi-head task calibration.

---

## 4. Application to Pretraining vs. Fine-Tuning

| Calibration Component | Stage 1: Synthetic Pretraining | Stage 2: Real-World Fine-Tuning | Rationale |
| :--- | :--- | :--- | :--- |
| **Inter-Head Task Loss ($\boldsymbol{\lambda}$)** | **Applied** ($[1.0, 1.25, 0.8, 0.5, 0.3, 0.2]$) | **Applied** ($[1.0, 1.25, 0.8, 0.5, 0.3, 0.2]$) | Boosts bass ($1.25\times$) and stabilizes Conformer backbone from Day 1 across both stages. |
| **Intra-Head Class Reweighting ($\alpha, C_{\max}$)** | **Uniform** ($w_c = 1.0$) | **Applied** ($\alpha=0.25, C_{\max}=4.0$) | Synthetic corpus is quota-balanced by `target_corpus_gen.py`. Real fine-tuning counters empirical Zipfian long-tail. |
| **`b7_clamp` Deactivation** | **Active** (No clamp) | **Active** (No clamp) | Prevents dominant 7th from dominating gradient updates in either stage. |
| **Head 1 (Bass) Key Balancing** | Natural synthetic keys | **Applied** (13-class empirical weights) | Prevents overfitting to guitar-centric keys in real pop/rock. |

---

## 5. Acceptance Criteria & Benchmark Targets

When evaluating on the 1,217-song benchmark test set (`eval_heads.py` with CRF penalty 2.0, tau 0.0):

| Head / Metric | Uncalibrated Baseline | Uncalibrated Synth FT | Calibrated Target | Success Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Head 0 (Triad Frame Acc)** | 0.7497 | 0.7319 | $\ge \mathbf{0.7520}$ | Frame accuracy exceeds baseline |
| **Head 1 (Bass Frame Acc)** | 0.8537 | 0.8376 | $\ge \mathbf{0.8550}$ | Recovers the $-1.61\%$ deficit via $1.25\times$ weight |
| **Head 2 (7th Frame Acc)** | 0.7718 | 0.7496 | $\ge \mathbf{0.7780}$ | Recovers the $-2.22\%$ deficit via `b7_clamp` removal |
| **`b7` False Positive Rate on `N`** | — | 19.52% (1.55M frames) | $\le \mathbf{7.5\%}$ | Resolves false-positive flood on triads |
| **`sus4` Precision** | 0.3620 | 0.2796 | $\ge \mathbf{0.4200}$ | Eliminates 247k false positives from `maj` |
| **`5` (Power Chord) Precision** | 0.3150 | 0.2559 | $\ge \mathbf{0.3800}$ | Eliminates 168k false positives from `maj` |
| **Head 4 (11th) Precision** | 0.1650 | 0.1266 | $\ge \mathbf{0.2500}$ | Reduces 202k false positives from `N` |
| **Altered Extensions Recall** | Baseline | Retained gains | **Sustained** | Retains positive recall gains on `#9, b9, #11, b13` |
