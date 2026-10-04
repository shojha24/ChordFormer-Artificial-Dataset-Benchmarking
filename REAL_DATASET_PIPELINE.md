# ChordFormer Real Dataset Pipeline Guide
**Non-Artificial 1217 Corpus: Storage Preparation → Training → Evaluation**

This guide provides the complete execution procedure for training and evaluating ChordFormer exclusively on the **real 1217 music corpus** (no synthetic or artificial data). It covers storage creation, both the original baseline `(1.0, 1.0)` and the corrected reweighted `(0.5, 10.0)` training regimes, 5-fold cross-validation, and multi-head evaluation.

---

## 0. Prerequisites & Environment

```bash
conda activate chordformer-bench
export CHORDFORMER_GPU=0   # Optional: set target GPU index
```

* **Working root:** `/home/shojha/ChordFormer-Artificial-Dataset-Benchmarking`
* **Real Dataset Source:** 1,217 songs spanning Billboard, Beatles, Queen, Robbie Williams, and Zweieck datasets registered in `new_datasets.py`.

---

## Real Pipeline Overview

```
                      [ Raw Real 1217 Audio & Chord Annotations ]
                                         │
                                         ▼
                                storage_creation.py
                                         │
                        ┌────────────────┴────────────────┐
                        ▼                                 ▼
             dataset/jams_cqt.h5d              dataset/jams_xchord.h5d
             (252-bin CQT features)            (6-head ground-truth labels)
                        │                                 │
                        └────────────────┬────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
                   ▼ (Regime A: Baseline)                      ▼ (Regime B: Corrected Reweighted)
      python confor_head16.py {0..4}              python confor_head16.py {0..4}
      --power 1.0 --max_clip 1.0                  --power 0.5 --max_clip 10.0
      (Pure Unweighted Cross-Entropy)             (Rare classes boosted up to 10x)
                   │                                           │
                   ▼                                           ▼
      cache_data/chordformer_head16(1.0,1.0)_s*.best.sdict   cache_data/chordformer_head16(0.5,10.0)_s*.best.sdict
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                                   eval_heads.py
                       (CRF Viterbi Decoding on Test Splits)
```

---

## Step 1: Build Real (1217) Dataset Storage

Generate the HDF5 caches for CQT spectrograms, 6-head chord labels, and 5-fold cross-validation weights:

```bash
python storage_creation.py
```

### Outputs Created:
* `dataset/jams_cqt.h5d`: Constant-Q Transform features (252 frequency bins, 24 bins/octave, hop length 4096 at 44.1 kHz).
* `dataset/jams_xchord.h5d`: Frame-aligned 6-head label matrix:
  1. `Head 0`: Root + Triad quality (97 classes: 1 'N' + 8 qualities $\times$ 12 roots)
  2. `Head 1`: Bass pitch class (13 classes: 1 'N' + 12 pitch classes)
  3. `Head 2`: 7th Extension (4 classes: `N`, `7`, `b7`, `bb7`)
  4. `Head 3`: 9th Extension (4 classes: `N`, `9`, `#9`, `b9`)
  5. `Head 4`: 11th Extension (3 classes: `N`, `11`, `#11`)
  6. `Head 5`: 13th Extension (3 classes: `N`, `13`, `b13`)
* `data/cross_subpart_weight{0..4}.pkl`: Class frequency counters for each of the 5 cross-validation splits.

---

## Step 2: Model Training on Real Music

Two training configurations are supported via `confor_head16.py`:

### Regime A: Original Baseline `(1.0, 1.0)`
Due to the `max_clip = 1.0` bug, every class weight is clipped to $1.0$, resulting in standard unweighted cross-entropy across all 6 heads:

```bash
# Single fold (e.g. Fold 0)
python confor_head16.py 0 --power 1.0 --max_clip 1.0

# All 5 folds sequentially:
for fold in 0 1 2 3 4; do
  python confor_head16.py $fold --power 1.0 --max_clip 1.0
done
```
* **Saved Checkpoint:** `cache_data/chordformer_head16(1.0,1.0)_s{0..4}.best.sdict`
* **Behavior:** High accuracy on common major/minor triads, but **0.0000 recall** on rare altered extensions (`#9`, `b9`, `b13`) because `'N'` frames overpower rare classes by 1,000:1.

---

### Regime B: Corrected Real Reweighting `(0.5, 10.0)`
Applies square-root inverse frequency weighting (`power=0.5`) capped at a 10x ceiling (`max_clip=10.0`). Rare extensions receive up to a 10x loss gradient penalty without numerical instability:

```bash
# Single fold (e.g. Fold 0)
python confor_head16.py 0 --power 0.5 --max_clip 10.0

# All 5 folds sequentially:
for fold in 0 1 2 3 4; do
  python confor_head16.py $fold --power 0.5 --max_clip 10.0
done
```

* **Saved Checkpoint:** `cache_data/chordformer_head16(0.5,10.0)_s{0..4}.best.sdict`
* **Optional Flags:**
  * `--val_metric macro`: Saves `.best.sdict` based on macro class accuracy across heads rather than purely frame-wise loss.
  * `--batch_size 48`: Training mini-batch size (default: 48).
  * `--num_workers 8`: Multi-processing worker threads.

---

## Step 3: Evaluation with CRF Viterbi Decoding

Evaluate the trained 5-fold models across all 6 heads on the held-out test splits using Conditional Random Field (CRF) temporal smoothing ($\tau = 2.0$):

### 3.1 Evaluate Baseline `(1.0, 1.0)` Models:
```bash
python eval_heads.py \
  --model_pattern "chordformer_head16(1.0,1.0)_s%d.best" \
  --crf_penalty 2.0 \
  --save_path eval_heads_baseline_report.txt
```

### 3.2 Evaluate Corrected Reweighted `(0.5, 10.0)` Models:
```bash
python eval_heads.py \
  --model_pattern "chordformer_head16(0.5,10.0)_s%d.best" \
  --crf_penalty 2.0 \
  --save_path eval_heads_reweighted_report.txt
```

---

## Step 4: Compare Performance Gains

Compare class-wise accuracy and rare chord recall between any two trained models:

```bash
python evaluate_synth_vs_baseline.py \
  --baseline eval_heads_baseline_report.txt \
  --synth eval_heads_reweighted_report.txt \
  --out comparison_reweighted_vs_baseline.txt
```

---

## Quick-Start Cheat Sheet

```bash
# 1. Build real HDF5 dataset storage
python storage_creation.py

# 2. Train real reweighted model (0.5 power, 10 max_clip) on Fold 0
python confor_head16.py 0 --power 0.5 --max_clip 10.0

# 3. Evaluate 5-fold models
python eval_heads.py \
  --model_pattern "chordformer_head16(0.5,10.0)_s%d.best" \
  --crf_penalty 2.0 \
  --save_path eval_heads_reweighted_report.txt
```
