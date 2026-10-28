# ChordFormer ACR: End-to-End Pipeline Guide
**Synthetic Pretraining → Real Domain Fine-Tuning → Multi-Head Evaluation**

This guide provides the complete, step-by-step instructions to reproduce the ChordFormer Automatic Chord Recognition (ACR) training pipeline from scratch: preparing real (1217) and synthetic (5,000-song) storage datasets, running Stage 1 synthetic pretraining, Stage 2 real fine-tuning with replay and discriminative learning rates, and executing evaluation benchmarks across all 6 chord heads.

---

## 0. Prerequisites & Environment

Activate the dedicated Conda environment and verify GPU availability:

```bash
conda activate chordformer-bench
export CHORDFORMER_GPU=0   # Optional: set target GPU index (0 by default)
```

Ensure the working directories and datasets are present:
* Working root: `/home/shojha/ChordFormer-Artificial-Dataset-Benchmarking`
* Synthetic audio/labels: `/home/shojha/probabilistic-composition-generator`
  * Audio: `gen/acr-conditions/naturalistic/audio/song_*/mix.flac`
  * Pop/Rock labels: `gen/acr-target-500k/pop-rock-labels/song_*.json` (0..3499)
  * Jazz labels: `gen/acr-target-500k/jazz-labels/song_*.json` (0..1499)

---

## Pipeline Overview

```
[ Step 1: Real 1217 Storage ]            [ Step 2: Synthetic Storage ]
       storage_creation.py                      convert_synth_labels.py
        │                                        create_synth_storage.py
        ▼                                                │
   dataset/jams_cqt.h5d                                  ▼
   dataset/jams_xchord.h5d                     dataset/synth_naturalistic_cqt.h5d
                                               dataset/synth_naturalistic_xchord.h5d
        │                                                │
        │                        ┌───────────────────────┘
        │                        │ (Stage 1 Pretraining)
        │                        ▼
        │              pretrain_synthetic.py
        │              (SpecAugment + uniform harmonic space)
        │                        │
        │                        ▼
        │              cache_data/chordformer_head16_synth_pretrain.best.sdict
        │                        │
        └───────────────┬────────┘
                        │ (Stage 2 Fine-Tuning: Replay + Discriminative LR)
                        ▼
               finetune_real_fold.py {0..4}
               - 80% Real / 20% Synth Replay
               - Corrected ReweightedLoss (power=0.5, max_clip=10.0)
               - Discriminative LR (Bottom: 1.0x, Top: 0.5x, Head: 0.1x)
               - Macro-score checkpoint selection
                        │
                        ▼
               cache_data/chordformer_head16_synth_ft_s{0..4}.best.sdict
                        │
                        ▼ (Stage 3 Evaluation)
               eval_heads.py (CRF Viterbi Decoding across all 6 heads)
               evaluate_synth_vs_baseline.py (Gain vs. Real Baseline)
               val_chordformer_head16.py (HMM Decoding to Harte .lab)
               test_for_all_mirex.py (mir_eval MIREX / WCSR Duration-Weighted Recall)
```

---

## Step 1: Build Real (1217) Dataset Storage

If `dataset/jams_cqt.h5d` and `dataset/jams_xchord.h5d` do not exist, generate them:

```bash
python storage_creation.py
```

* **Output files created:**
  * `dataset/jams_cqt.h5d` (252-bin CQT spectrogram features for 1,217 songs)
  * `dataset/jams_xchord.h5d` (6-head ground-truth chord label matrices)
  * `data/cross_subpart_weight{0..4}.pkl` (Class frequency counters for 5-fold cross-validation)

---

## Step 2: Build Artificial Dataset Storage

### 2.1 Convert Synthetic JSON Annotations to Harte `.lab` Files

Converts 5,000 synthetic song JSONs (3,500 Pop/Rock + 1,500 Jazz, matching the 70/30 genre split) into Harte `.lab` format with chord syntax normalization:

```bash
python convert_synth_labels.py
```
* **Output:** 5,000 files in `data/synth_naturalistic_lab/song_0.lab` ... `song_4999.lab`.

### 2.2 Extract CQTs and Build Synthetic HDF5 Storage

Extracts 252-bin Constant-Q Transforms across 16 parallel workers and caches features into HDF5:

```bash
python create_synth_storage.py
```
* **Output files created:**
  * `dataset/synth_naturalistic_cqt.h5d` (5,000 synthetic audio CQTs, ~20 GB)
  * `dataset/synth_naturalistic_xchord.h5d` (5,000 6-head label matrices, ~425 MB)

---

## Step 3: Stage 1 — Synthetic Pretraining

Pretrains the Conformer backbone and 6 output heads over the synthetic harmonic distribution using SpecAugment (Time & Frequency masking) to encourage robust invariant representations:

```bash
python pretrain_synthetic.py \
  --batch_size 36 \
  --num_workers 8 \
  --save_name chordformer_head16_synth_pretrain
```

* **Default LR Schedule:** 15 epochs at $10^{-3}$, 10 epochs at $10^{-4}$, 5 epochs at $10^{-5}$.
* **Data Split:** 4,500 training songs / 500 validation songs.
* **Output Checkpoint:** `cache_data/chordformer_head16_synth_pretrain.best.sdict`.

---

## Step 4: Stage 2 — Real Domain Fine-Tuning

Fine-tunes the pretrained model on real music folds using the 4-part anti-forgetting architecture:
1. **Replay Buffer (`--replay_ratio 0.2`):** Mixes 80% real audio with 20% synthetic replay samples per batch.
2. **Loss Reweighting (`--loss_power 0.5 --loss_max_clip 10.0`):** Gives rare alterations up to 10x weight without gradient explosion.
3. **Discriminative LR:** Lower layers adapt to real acoustic timbre ($1.0 \times \text{lr}$), upper layers preserve contextual grammar ($0.5 \times \text{lr}$), and classification heads retain rare extension weights ($0.1 \times \text{lr}$).
4. **Macro Checkpointing (`--val_metric macro`):** Selects best checkpoints based on macro class accuracy across all 6 heads.

### Run a Single Fold (e.g. Fold 0):
```bash
python finetune_real_fold.py 0 \
  --pretrained_path cache_data/chordformer_head16_synth_pretrain.best.sdict \
  --save_prefix chordformer_head16_synth_ft \
  --replay_ratio 0.2 \
  --loss_max_clip 10.0 \
  --loss_power 0.5 \
  --val_metric macro
```

### Run All 5 Folds:
```bash
python finetune_real_fold.py -1 \
  --pretrained_path cache_data/chordformer_head16_synth_pretrain.best.sdict \
  --save_prefix chordformer_head16_synth_ft \
  --replay_ratio 0.2 \
  --loss_max_clip 10.0 \
  --loss_power 0.5 \
  --val_metric macro
```

* **Output Checkpoints:** `cache_data/chordformer_head16_synth_ft_s0.best.sdict` ... `s4.best.sdict`.

---

## Step 5: Stage 3 — Evaluation & Benchmarking

### 5.1 Evaluate Fine-Tuned Model Across All 6 Heads
Runs CRF Viterbi decoding over all 5 cross-validation test splits:

```bash
python eval_heads.py \
  --model_pattern "chordformer_head16_synth_ft_s%d.best" \
  --crf_penalty 2.0 \
  --save_path eval_heads_synth_ft_report.txt
```

### 5.2 Compare Against Real Baseline
Calculates macro gain, head frame accuracy, and rare alteration recall gains against the baseline 1217 model:

```bash
python evaluate_synth_vs_baseline.py \
  --baseline eval_heads_report.txt \
  --synth eval_heads_synth_ft_report.txt \
  --out comparison_summary.txt
```

### 5.3 Standard MIREX / WCSR Evaluation (Segment-Level Duration-Weighted Recall)

To evaluate according to standard MIREX Automatic Chord Recognition (ACR) conventions using Weighted Chord Symbol Recall (WCSR):

#### 1. Decode Test Tracks with HMM to Harte `.lab` Files:
Inference runs across the 5 cross-validation folds on the 1,217 real songs, decoding multi-head frame predictions into standard chord interval annotations via `XHMMDecoder`. Decoding is parallelized across CPU cores (`--n_jobs 12` by default):

```bash
python val_chordformer_head16.py "<model_pattern>" <chord_dict> [--n_jobs 12]
```

* **Reweighted (0.5, 10.0) Checkpoints:**
  ```bash
  python val_chordformer_head16.py "chordformer_head16(0.5,10.0)_s%d.best" full --n_jobs 12
  ```
* **Fine-Tuned Checkpoints:**
  ```bash
  python val_chordformer_head16.py "chordformer_head16_synth_ft_s%d.best" full --n_jobs 12
  ```
* **Baseline Real 1217 Checkpoints:**
  ```bash
  python val_chordformer_head16.py "chordformer_head16(1.0,1.0)_s%d.best" full --n_jobs 12
  ```
* **Output:** Generates 1,217 `.lab` files under `output/output_<model_pattern>_hmm_<chord_dict>/jam/*.lab`.

#### 2. Compute Duration-Weighted MIREX / WCSR Accuracies:
Evaluates the decoded `.lab` files against the ground-truth annotations in `chord_data_1217/chordlab/` using `mir_eval.chord.weighted_accuracy`:

```bash
python test_for_all_mirex.py "output/<output_folder>/jam/"
```

* **Example for reweighted (0.5, 10.0) models:**
  ```bash
  python test_for_all_mirex.py "output/output_chordformer_head16(0.5,10.0)_s%d.best_hmm_full/jam/"
  ```
* **Example for fine-tuned models:**
  ```bash
  python test_for_all_mirex.py "output/output_chordformer_head16_synth_ft_s%d.best_hmm_full/jam/"
  ```
* **Example for baseline 1217 models:**
  ```bash
  python test_for_all_mirex.py "output/output_chordformer_head16(1.0,1.0)_s%d.best_hmm_full/jam/"
  ```
* **Metrics Computed:**
  * **WCSR Scores (%):** Root, Thirds, MajorMinor, Triads, Sevenths, Tetrads, MIREX, and bass-inversions (`thirds_inv`, `majmin_inv`, `triads_inv`, `sevenths_inv`, `tetrads_inv`).
  * **Detailed Quality Breakdown:** Per-class Ground Truth Duration (s), GT Share, Recall, Precision, F1-Score, and Top Confusion.
  * **Output Artifacts:** Writes `eval_results.txt`, `eval_detailed_breakdown.txt`, `confusion_matrix.pdf`, and `confusion_matrix.png`.

---

## Quick-Start Command Cheat Sheet

```bash
# 1. Prepare synthetic storage
python convert_synth_labels.py
python create_synth_storage.py

# 2. Stage 1: Pretrain on synthetic data
python pretrain_synthetic.py --batch_size 36

# 3. Stage 2: Fine-tune on real folds (Fold 0 test)
python finetune_real_fold.py 0 --replay_ratio 0.2 --loss_max_clip 10.0 --val_metric macro

# 4. Stage 3: Evaluate and compare (Frame-Level CRF)
python eval_heads.py --model_pattern "chordformer_head16_synth_ft_s%d.best" --save_path eval_heads_synth_ft_report.txt
python evaluate_synth_vs_baseline.py --baseline eval_heads_report.txt --synth eval_heads_synth_ft_report.txt

# 5. Stage 4: Standard MIREX / WCSR Duration-Weighted Evaluation
python val_chordformer_head16.py "chordformer_head16_synth_ft_s%d.best" full --n_jobs 12
python test_for_all_mirex.py "output/output_chordformer_head16_synth_ft_s%d.best_hmm_full/jam/"
```
