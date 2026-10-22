# ChordFormer ACR: End-to-End Pipeline Guide
**Synthetic Pretraining (5,000 Songs) → Real Domain Fine-Tuning → Multi-Head Evaluation**

This guide provides complete, step-by-step instructions to reproduce the ChordFormer Automatic Chord Recognition (ACR) training pipeline from scratch: preparing real (1,217 songs) and synthetic (5,000 songs / ~500,000 chord events, Spec 18) storage datasets, running Stage 1 synthetic pretraining with SpecAugment and calibrated 7-epoch schedule, Stage 2 real fine-tuning with floor curriculum replay (15% floor) and inverted discriminative learning rates, and executing evaluation benchmarks across all 6 chord heads with fast CRF Viterbi decoding.

---

## 0. Prerequisites & Environment

Activate the dedicated Conda environment and verify GPU availability:

```bash
conda activate chordformer-bench
export CHORDFORMER_GPU=0   # Optional: set target GPU index (0 by default)
```

Ensure the working directories and generated 5k assets (or 10k assets) are present:
* **Working root:** `/home/shojha/ChordFormer-Artificial-Dataset-Benchmarking`
* **Synthetic audio & labels:** `/home/shojha/probabilistic-composition-generator`
  * Audio: `gen/acr-conditions-5k/naturalistic/audio/song_*/mix.flac` (0..4999) (or `acr-conditions-10k`)
  * Pop/Rock labels: `gen/acr-target-5k/pop-rock-labels/song_*.json` (0..3249)
  * Jazz labels: `gen/acr-target-5k/jazz-labels/song_*.json` (0..1749)

---

## Pipeline Overview

```
[ Step 1: Real 1217 Storage ]            [ Step 2: 5,000-Song Synthetic Storage ]
       storage_creation.py                       convert_synth_labels.py --total_songs 5000 --pop_count 3250
        │                                         create_synth_storage.py --overwrite --num_songs 5000
        ▼                                                 │
   dataset/jams_cqt.h5d                                   ▼
   dataset/jams_xchord.h5d                      dataset/synth_naturalistic_cqt.h5d (~22 GB)
                                                dataset/synth_naturalistic_xchord.h5d (~425 MB)
        │                                                 │
        │                        ┌────────────────────────┘
        │                        │ (Stage 1 Pretraining: ~10,500 steps / 7 epochs)
        │                        ▼
        │              pretrain_synthetic.py --total_songs 5000
        │              - SpecAugment (Time & Frequency masking)
        │              - 4,500 train songs / 500 held-out val songs
        │              - Calibrated LR schedule ({1e-3: 4, 1e-4: 2, 1e-5: 1})
        │                        │
        │                        ▼
        │              cache_data/chordformer_head16_synth_pretrain.best.sdict
        │                        │
        └───────────────┬────────┘
                        │ (Stage 2 Fine-Tuning: Floor Curriculum Replay + Inverted Discrim LR)
                        ▼
               finetune_real_fold.py {0..4} --no_load_checkpoint
               - Floor Curriculum Replay: 20% (1e-4) → 17.5% (1e-5) → 15% floor (1e-6)
               - Corrected ReweightedLoss (power=0.5, max_clip=10.0, b7_clamp=True)
               - Inverted Discriminative LR (Bottom: 0.2x, Top: 0.5x, Head: 1.0x)
               - Macro-score checkpoint selection across 6 heads
                        │
                        ▼
               cache_data/chordformer_head16_synth_ft_s{0..4}.best.sdict
                        │
                        ▼ (Stage 3 Evaluation & Benchmarking)
               eval_heads.py
               - 400x Fast CRF Viterbi Decoding across all 6 heads
               - Post-Hoc Logit Adjustment (tau=0.0 / --no_logit_adj recommended for rare recall)
               evaluate_synth_vs_baseline.py (Gain vs. Real Baseline)
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
  * `data/cross_subpart_weight{0..4}.pkl` (Empirical class frequency counters for 5-fold cross-validation)

---

## Step 2: Build Artificial Dataset Storage (5,000 Songs Standard)

### 2.1 Convert Synthetic JSON Annotations to Harte `.lab` Files

Converts 5,000 synthetic song JSONs (3,250 Pop/Rock + 1,750 Jazz, matching the 65/35 genre split) into Harte `.lab` format with syntax normalization and slash-chord support:

```bash
python convert_synth_labels.py --total_songs 5000 --pop_count 3250
```
* **Output:** 5,000 files in `data/synth_naturalistic_lab/song_0.lab` ... `song_4999.lab`.
* *(Note: If using a 10,000-song corpus, pass `--total_songs 10000 --pop_count 6500`).*

### 2.2 Extract CQTs and Build Synthetic HDF5 Storage

Extracts 252-bin Constant-Q Transforms across 16 parallel workers and caches features into HDF5 datasets:

```bash
python create_synth_storage.py --overwrite --num_songs 5000 --num_workers 16
```
* **Output files created:**
  * `dataset/synth_naturalistic_cqt.h5d` (5,000 synthetic audio CQTs, ~20–22 GB)
  * `dataset/synth_naturalistic_xchord.h5d` (5,000 6-head label matrices, ~425 MB)

---

## Step 3: Stage 1 — Synthetic Pretraining

Pretrains the Conformer backbone and 6 output heads over the synthetic harmonic distribution using SpecAugment (Time & Frequency masking) and pitch-shifting:

```bash
python pretrain_synthetic.py \
  --batch_size 36 \
  --num_workers 8 \
  --total_songs 5000 \
  --save_name chordformer_head16_synth_pretrain
```

* **Dataset Split:** 4,500 train / 500 held-out val songs (for 5k corpus).
* **Calibrated Schedule (Spec 18):**
  * **5,000 Songs (Standard):** 4 epochs at $10^{-3}$, 2 epochs at $10^{-4}$, 1 epoch at $10^{-5}$ ($\sim 10,500$ steps total; $\lceil \frac{4,500 \times 12}{36} \rceil = 1,500$ steps/epoch). Prevents synthetic timbre rigidity while delivering full harmonic coverage.
  * **10,000 Songs (Optional):** 3 epochs at $10^{-3}$, 2 epochs at $10^{-4}$, 1 epoch at $10^{-5}$ ($\sim 19,000$ steps total; pass `--total_songs 10000`).
  * Fine-grained override flags: `--epochs_1e3`, `--epochs_1e4`, `--epochs_1e5`.
* **Output Checkpoint:** `cache_data/chordformer_head16_synth_pretrain.best.sdict`.

---

## Step 4: Stage 2 — Real Domain Fine-Tuning

Fine-tunes the pretrained model on real music folds using the anti-forgetting architecture:
1. **Floor Curriculum Replay (`--replay_ratio 0.2 --replay_floor 0.15`):**
   * **Phase 1 ($10^{-4}$ LR, epochs 1–12):** `20%` synthetic replay (grounds rare extension features).
   * **Phase 2 ($10^{-5}$ LR, epochs 13–20):** `17.5%` synthetic replay (midpoint between base and floor).
   * **Phase 3 ($10^{-6}$ LR, epochs 21–24):** `15%` floor synthetic replay (never 0%, completely preventing catastrophic forgetting of altered chords).
   * Pass `--no_anneal_replay` if you wish to disable curriculum annealing and keep a flat 20%.
2. **Inverted Discriminative LR:**
   * Bottom Conformer layers: $0.2 \times \text{lr}$ (preserves pretrained acoustic filters and vocal formant invariance).
   * Upper Conformer layers: $0.5 \times \text{lr}$.
   * Output classification heads: $1.0 \times \text{lr}$ (adapts rapidly to real class distributions).
3. **Loss Reweighting (`--loss_power 0.5 --loss_max_clip 10.0`):** Weights rare chord alterations up to 10x without gradient explosion, with rare-class-safe floor protection for intermediate classes (`b7`, `sus4`, `5`, `min`, `9`).
4. **Macro Checkpointing (`--val_metric macro`):** Selects best checkpoints based on macro class accuracy across all 6 heads.

### Run a Single Fold (e.g. Fold 0):
```bash
python finetune_real_fold.py 0 \
  --no_load_checkpoint \
  --pretrained_path cache_data/chordformer_head16_synth_pretrain.best.sdict \
  --save_prefix chordformer_head16_synth_ft \
  --replay_ratio 0.20 \
  --replay_floor 0.15 \
  --loss_max_clip 10.0 \
  --loss_power 0.5 \
  --val_metric macro
```

### Run All 5 Cross-Validation Folds:
```bash
python finetune_real_fold.py -1 \
  --no_load_checkpoint \
  --pretrained_path cache_data/chordformer_head16_synth_pretrain.best.sdict \
  --save_prefix chordformer_head16_synth_ft \
  --replay_ratio 0.20 \
  --replay_floor 0.15 \
  --loss_max_clip 10.0 \
  --loss_power 0.5 \
  --val_metric macro
```

### Resume or Run Specific Folds (e.g. Folds 3 and 4):
```bash
# Option 1: Pass comma-separated folds (e.g. 3,4 or 3-4)
python finetune_real_fold.py 3,4 \
  --no_load_checkpoint \
  --pretrained_path cache_data/chordformer_head16_synth_pretrain.best.sdict \
  --save_prefix chordformer_head16_synth_ft \
  --replay_ratio 0.20 \
  --replay_floor 0.15 \
  --loss_max_clip 10.0 \
  --loss_power 0.5 \
  --val_metric macro

# Option 2: Use -1 with --start_fold 3 (skips folds 0, 1, 2)
python finetune_real_fold.py -1 --start_fold 3 \
  --no_load_checkpoint \
  --pretrained_path cache_data/chordformer_head16_synth_pretrain.best.sdict \
  --save_prefix chordformer_head16_synth_ft \
  --replay_ratio 0.20 \
  --replay_floor 0.15 \
  --loss_max_clip 10.0 \
  --loss_power 0.5 \
  --val_metric macro
```

* **Output Checkpoints:** `cache_data/chordformer_head16_synth_ft_s0.best.sdict` ... `s4.best.sdict`.

---

## Step 5: Stage 3 — Evaluation & Benchmarking

### 5.1 Evaluate Fine-Tuned Model Across All 6 Heads
Runs CRF Viterbi decoding and **Post-Hoc Logit Adjustment** ($\tau = 0.35$ by default) across all 5 test splits (1,217 songs):

```bash
python eval_heads.py \
  --model_pattern "chordformer_head16_synth_ft_s%d.best" \
  --crf_penalty 2.0 \
  --tau 0.0 \
  --save_path eval_heads_synth_ft_report.txt
```

* **Logit Adjustment Options:**
  * `--tau 0.35` (default): Adjusts inference logits $z'_k = z_k + \tau \log \pi_k$ using empirical natural class priors, eliminating false positives on rare extensions and recovering overall frame accuracy.
  * `--no_logit_adj` or `--tau 0.0`: Runs raw unadjusted inference.

### 5.2 Compare Against Real Baseline
Calculates macro gain, head frame accuracy, and rare alteration recall gains against the baseline 1217 model:

```bash
python evaluate_synth_vs_baseline.py \
  --baseline eval_heads_report.txt \
  --synth eval_heads_synth_ft_report.txt \
  --out eval_comparison_summary.txt
```

---

## Quick-Start Command Cheat Sheet (5k Dataset Standard)

```bash
# 1. Prepare synthetic labels & HDF5 storage (5,000 songs)
python convert_synth_labels.py
python create_synth_storage.py --overwrite --num_workers 16

# 2. Stage 1: Pretrain on synthetic 5k corpus (~10,500 steps, 7 epochs)
python pretrain_synthetic.py

# 3. Stage 2: Fine-tune on real folds (Fold 0 test, warm-starting from Stage 1 with 15% floor replay)
python finetune_real_fold.py 0 --no_load_checkpoint --replay_ratio 0.2 --replay_floor 0.15 --val_metric macro

# 4. Stage 3: Evaluate with 400x Fast CRF (--tau 0.0 for maximum rare recall) and compare
python eval_heads.py --model_pattern "chordformer_head16_synth_ft_s%d.best" --tau 0.0 --save_path eval_heads_synth_ft_report.txt
python evaluate_synth_vs_baseline.py --baseline eval_heads_report.txt --synth eval_heads_synth_ft_report.txt
```
