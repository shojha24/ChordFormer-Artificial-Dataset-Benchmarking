# ChordFormer: Sim-to-Real Synthetic Pretraining & Benchmarking

> **Preprint Companion Repository:** *Distribution-Shifted Synthetic Multitrack Pretraining for Large-Vocabulary Audio Chord Recognition* (Sharabh Ojha, UCLA)  
> **Companion Synthetic Audio Generator Repository:** [`probabilistic-composition-generator`](https://github.com/shojha24/probabilistic-composition-generator)

ChordFormer is a Conformer-based audio chord recognition (ACR) system for large vocabularies (`confor_head16.py`, $|\Theta| = 6{,}123{,}644$ parameters). Whereas the original ChordFormer (`triad_limit=6`, 100 logits) restricts Head 0 to 6 triad qualities and collapses power chords (`5`) and bare-root unisons (`1`) into the major triad (`maj`), we expand Head 0 to 8 triad qualities (`triad_limit=8`, 97 Head-0 logits / 124 total logits) by leveraging standalone synthetic training exemplars for `5` and `1`. This repository contains the implementation of our **two-stage Sim-to-Real curriculum**—pretraining on 5,000 distribution-shifted procedural multitrack recordings ($460.3\,\text{h}$) followed by 5-fold cross-validated finetuning on the 1,217-song real benchmark ($73.7\,\text{h}$, $11{,}436{,}357$ frames) with inverted discriminative learning rates ($\mathbf{s}_{\text{lr}} = [0.2, 0.5, 1.0]$) and floor curriculum replay ($\rho_{\min} = 0.10$)—as well as our **FP16 ONNX/TensorRT 10.x GKE Triton inference pipeline**.

---

## 5-Fold Cross-Validation Benchmark Results (`s0`–`s4`, 1,217 Real Songs)

We compare two regimens under identical CQT preprocessing ($f_s = 22{,}050\,\text{Hz}$, $H = 512$, $F = 252$ bins across 7 octaves padded to 256) and inverse-frequency reweighted cross-entropy ($\alpha = 0.50, C_{\max} = 10.0$):
- **Control Baseline (`control-benchmark`):** Trained from scratch on the 1,217 real songs (24 epochs/fold, $B=36/48$, $\eta \in \{10^{-3}\!:\!12, 10^{-4}\!:\!8, 10^{-5}\!:\!4\}$).
- **Experimental Sim-to-Real Curriculum (`finetuned-benchmark`):** Pretrained in Stage 1 on 5,000 synthetic random-mix tracks (7 epochs, $10{,}500$ mini-batches, SpecAugment + pitch shifting) and finetuned in Stage 2 on each real fold (24 epochs, $B=36$, $\eta \in \{10^{-4}\!:\!12, 10^{-5}\!:\!8, 10^{-6}\!:\!4\}$, inverted discriminative LR $\mathbf{s}_{\text{lr}} = [0.2, 0.5, 1.0]$, and floor curriculum replay $\rho(\eta) = 0.150 \to 0.125 \to 0.100$).

### 1. Structured 6-Head Decomposition & Rare Triad Performance ($\tau_{\text{crf}} = 2.0$)

| Structural Output Head | Control Recall (%) | Experimental Recall (%) | $\Delta$ Rec. (pp) | Control $F_1$ (%) | Experimental $F_1$ (%) | $\Delta$ $F_1$ (pp) |
|---|---:|---:|---:|---:|---:|---:|
| **Root + Triad (97 classes)** | $41.21 \pm 3.12$ | **$42.66 \pm 2.59$** | **$+1.45$** | $36.57 \pm 2.07$ | **$36.70 \pm 1.68$** | **$+0.13$** |
| **Bass Pitch (13 classes)** | **$84.81 \pm 1.73$** | $82.93 \pm 1.51$ | $-1.88$ | **$84.72 \pm 1.77$** | $83.01 \pm 1.57$ | $-1.70$ |
| **7th Extension (4 classes)** | **$62.69 \pm 4.27$** | $62.55 \pm 3.14$ | $-0.14$ | **$57.67 \pm 2.73$** | $56.99 \pm 1.48$ | $-0.67$ |
| **9th Extension (4 classes)** | $35.33 \pm 2.68$ | **$43.58 \pm 7.54$** | **$+8.25$** | $34.98 \pm 2.96$ | **$39.01 \pm 6.25$** | **$+4.03$** |
| **11th Extension (3 classes)** | $38.74 \pm 7.85$ | **$41.71 \pm 8.03$** | **$+2.97$** | $37.35 \pm 3.60$ | **$42.14 \pm 7.04$** | **$+4.79$** |
| **13th Extension (3 classes)** | **$35.36 \pm 3.74$** | $34.26 \pm 1.95$ | $-1.10$ | **$36.09 \pm 4.82$** | $34.47 \pm 2.47$ | $-1.62$ |
| **Macro Average (6 Heads)** | $49.69 \pm 2.10$ | **$51.28 \pm 1.94$** | **$+1.59$** | $47.90 \pm 1.69$ | **$48.72 \pm 1.86$** | **$+0.82$** |
| *Triad Quality (all 9 classes)* | $50.23 \pm 4.53$ | **$53.42 \pm 3.96$** | **$+3.19$** | $45.52 \pm 3.40$ | **$46.91 \pm 2.57$** | **$+1.39$** |
| *Common Triads (`maj`, `min`)* | **$84.74 \pm 1.97$** | $83.08 \pm 1.44$ | $-1.66$ | **$86.99 \pm 1.19$** | $86.05 \pm 0.85$ | $-0.94$ |
| *Rare Triads (`sus2`, `sus4`, `dim`, `aug`)* | $41.67 \pm 7.30$ | **$48.31 \pm 6.47$** | **$+6.64$** | $53.11 \pm 8.62$ | **$60.94 \pm 6.26$** | **$+7.83$** |

**Headline Per-Class Rare Chord & Extension Gains (11,436,357 evaluated frames):**
- **Rare Triads & Power Chords:** `aug` recall **$43.37\% \to 61.02\%$ ($+17.65\,\text{pp}$**, $F_1$ $35.40\% \to 43.15\%$); `5` recall **$28.76\% \to 41.91\%$ ($+13.15\,\text{pp}$)**; `sus2` recall **$17.32\% \to 30.25\%$ ($+12.93\,\text{pp}$**, $F_1$ $17.40\% \to 19.90\%$); `sus4` recall **$51.19\% \to 53.72\%$ ($+2.53\,\text{pp}$)**; `dim` recall **$65.61\% \to 67.97\%$ ($+2.36\,\text{pp}$**, $F_1$ $41.65\% \to 47.79\%$).
- **Seventh & Upper/Altered Extensions:** `9` recall **$31.64\% \to 56.80\%$ ($+25.16\,\text{pp}$**, $F_1$ $31.06\% \to 36.19\%$); `b7` recall **$70.31\% \to 80.90\%$ ($+10.59\,\text{pp}$)**; `#11` recall **$14.15\% \to 23.02\%$ ($+8.87\,\text{pp}$**, $F_1$ $13.39\% \to 20.50\%$); `b9` recall **$18.39\% \to 24.71\%$ ($+6.32\,\text{pp}$**, $F_1$ $9.56\% \to 18.02\%$); `b13` recall **$0.84\% \to 6.58\%$ ($+5.74\,\text{pp}$, $7.8\times$**, $F_1$ $0.77\% \to 8.43\%$); `#9` recall **$0.07\% \to 5.53\%$ ($+5.46\,\text{pp}$, $79\times$**, $F_1$ $0.10\% \to 4.97\%$).

### 2. Standard `mir_eval` Frame-Level WCSR & MIREX Benchmark ($\tau_{\text{crf}} = 30.0$)

| `mir_eval` Comparison Rule | Control `control-benchmark` (%) | Experimental `finetuned-benchmark` (%) | $\Delta$ (pp) |
|---|---:|---:|---:|
| **Root Pitch Only** | **$83.77 \pm 1.88$** | $82.30 \pm 1.58$ | $-1.47$ |
| **Major / Minor** | **$79.78 \pm 2.38$** | $77.04 \pm 1.71$ | $-2.74$ |
| **Major / Minor + Inversions** | **$77.77 \pm 2.22$** | $75.17 \pm 1.71$ | $-2.60$ |
| **Triads (Full Quality)** | **$75.01 \pm 2.72$** | $72.88 \pm 1.97$ | $-2.13$ |
| **Triads + Inversions** | **$73.15 \pm 2.58$** | $71.13 \pm 1.95$ | $-2.02$ |
| **Sevenths (Standard)** | **$66.46 \pm 2.46$** | $62.46 \pm 1.73$ | $-4.00$ |
| **Sevenths + Inversions** | **$65.12 \pm 2.42$** | $61.18 \pm 1.72$ | $-3.94$ |
| **Tetrads (Extended)** | **$61.62 \pm 2.73$** | $58.20 \pm 1.74$ | $-3.42$ |
| **Tetrads + Inversions** | **$60.38 \pm 2.69$** | $57.02 \pm 1.75$ | $-3.36$ |
| **MIREX Metric (Overall)** | **$81.49 \pm 2.56$** | $79.59 \pm 1.85$ | $-1.90$ |

> **Note on WCSR vs. Macro Recall:** Because WCSR weights frame accuracy strictly by duration ($\sum_i d_i \mathbb{I}(\hat{C}_i == C_i)/\sum_i d_i$), it is dominated by `maj` ($44.93\%$ of real duration) and `min` ($11.95\%$). Expanding minority decision boundaries to achieve best-in-class recall on rare chords introduces slight false-positive pressure on ambiguous majority-triad frames while consistently reducing cross-fold standard deviation across all ten evaluation rules.

---

## End-to-End Reproducibility Guide

### (a) Stage 1: Synthetic Pretraining (5,000 Random-Mix Songs)

1. Convert the 5,000 synthetic song JSONs (3,250 Pop/Rock + 1,750 Jazz) from `probabilistic-composition-generator` into frame-exact Harte `.lab` files and extract 252-bin CQT HDF5 storage (`dataset/synth_naturalistic_cqt.h5d`, $21.8\,\text{GB}$):

```bash
python3 convert_synth_labels.py --total_songs 5000 --pop_count 3250
python3 create_synth_storage.py --overwrite --num_songs 5000 --num_workers 16
```

2. Run Stage 1 synthetic pretraining (`pretrain_synthetic.py`) for 7 epochs ($10{,}500$ mini-batches at $B=36$, 4,500 train / 500 val split, unweighted cross-entropy, $[-5, +6]$ semitone pitch shifting, and SpecAugment with 2 frequency masks $\le 24$ bins and 2 time masks $\le 60$ frames at $-80.0\,\text{dB}$):

```bash
python3 pretrain_synthetic.py \
  --batch_size 36 \
  --num_workers 8 \
  --total_songs 5000 \
  --save_name chordformer_head16_synth_pretrain
```

### (b) Stage 2: 5-Fold Real-Audio Finetuning (`s0`–`s4`)

1. Ensure the 1,217-song real CQT/label HDF5 stores (`dataset/jams_cqt.h5d`, `dataset/jams_xchord.h5d`) and fold frequency counters (`data/cross_subpart_weight{0..4}.pkl`) are built:

```bash
python3 storage_creation.py
```

2. Finetune across all 5 folds (`finetune_real_fold.py -1`) for 24 epochs/fold ($B=36$, $\eta \in \{10^{-4}\!:\!12, 10^{-5}\!:\!8, 10^{-6}\!:\!4\}$) with inverted discriminative learning rates ($\mathbf{s}_{\text{lr}} = [0.2, 0.5, 1.0]$), floor curriculum replay ($\rho(\eta) = 0.150 \to 0.125 \to 0.100$), and inverse-frequency reweighting ($\alpha = 0.50, C_{\max} = 10.0$):

```bash
python3 finetune_real_fold.py -1 \
  --pretrained_path cache_data/chordformer_head16_synth_pretrain.best.sdict \
  --save_prefix chordformer_head16_synth_ft_loss \
  --val_metric loss \
  --replay_ratio 0.15 \
  --replay_floor 0.10 \
  --loss_power 0.5 \
  --loss_max_clip 10.0 \
  --no_load_checkpoint
```

### (c) Multi-Head & `mir_eval` WCSR/MIREX Evaluation

1. **6-Head Structural & Per-Class Recall/F1 Evaluation (`eval_heads.py`, $\tau_{\text{crf}} = 2.0$):**

```bash
# Evaluate Control Baseline (control-benchmark) across 5 folds
python3 eval_heads.py \
  --model_pattern "chordformer_head16(1.0,1.0)_s%d.best" \
  --crf_penalty 2.0 \
  --tau 0.0 \
  --save_path eval_heads_report.txt

# Evaluate Experimental Curriculum (finetuned-benchmark) across 5 folds
python3 eval_heads.py \
  --model_pattern "chordformer_head16_synth_ft_loss_s%d.best" \
  --crf_penalty 2.0 \
  --tau 0.0 \
  --save_path eval_heads_synth_ft_report.txt

# Generate side-by-side comparison summary
python3 evaluate_synth_vs_baseline.py \
  --baseline eval_heads_report.txt \
  --synth eval_heads_synth_ft_report.txt \
  --out eval_comparison_summary.txt
```

2. **Standard `mir_eval` Duration-Weighted WCSR & MIREX Evaluation (`val_chordformer_head16.py` + `test_for_all_mirex.py`, $\tau_{\text{crf}} = 30.0$):**

```bash
# Decode 5-fold test splits into Harte .lab annotations via XHMMDecoder (extractors/xhmm_ismir.py)
python3 val_chordformer_head16.py "chordformer_head16(1.0,1.0)_s%d.best" full --n_jobs 12
python3 val_chordformer_head16.py "chordformer_head16_synth_ft_loss_s%d.best" full --n_jobs 12

# Compute duration-weighted WCSR across all 10 mir_eval rules
python3 test_for_all_mirex.py "output/output_chordformer_head16(1.0,1.0)_s%d.best_hmm_full/jam/"
python3 test_for_all_mirex.py "output/output_chordformer_head16_synth_ft_loss_s%d.best_hmm_full/jam/"
```

### (d) 5-Fold ONNX / TensorRT FP16 GKE Triton Inference Pipeline

Our 5-fold ensemble (`s0`–`s4`, $5 \times 6{,}123{,}644$ parameters) is fused into a single computational graph (`model_repository/chordformer_ensemble/`) averaging softmax posteriors across all 5 folds and compiled into an **FP16 TensorRT 10.x engine** (`model.plan`) served via NVIDIA Triton Inference Server on Google Cloud GKE (`g2-standard-8`, 1$\times$ NVIDIA L4 GPU with $24\,\text{GB}$ GDDR6):

- **Single-Request Latency ($L = 1$):** **$2.83\,\text{ms}$** pure GPU compute latency ($10.0\,\text{s}$ audio segment = $431$ CQT frames).
- **Peak Dynamic Batching Throughput ($L = 13$):** **$395.75\,\text{inferences/sec}$** (**$4.28\times$ throughput speedup** over $L=1$'s $92.42\,\text{inf/s}$), with **$10.93\,\text{ms}$** GPU compute latency and **$32.56\,\text{ms}$** end-to-end client round-trip latency (**sub-40ms** SLA).

```bash
# 1. Export the 5-fold ChordFormer ensemble to a fused dynamic-batch/dynamic-time ONNX graph
python3 - << 'PYEOF'
import torch
from confor_head16 import ChordNet

class FusedEnsemble(torch.nn.Module):
    def __init__(self, ckpt_paths):
        super().__init__()
        self.models = torch.nn.ModuleList()
        for p in ckpt_paths:
            m = ChordNet()
            state = torch.load(p, map_location="cpu")
            m.load_state_dict(state["model"] if "model" in state else state)
            m.eval()
            self.models.append(m)

    def forward(self, x):
        logits = [m(x) for m in self.models]
        return torch.stack(logits, dim=0).mean(dim=0)

ckpts = [f"cache_data/chordformer_head16_synth_ft_loss_s{i}.best.sdict" for i in range(5)]
ensemble = FusedEnsemble(ckpts).cuda().eval()
dummy = torch.randn(1, 431, 256, device="cuda")
torch.onnx.export(
    ensemble, dummy, "model_repository/chordformer_ensemble/1/model.onnx",
    input_names=["cqt_input"], output_names=["head_logits"],
    dynamic_axes={"cqt_input": {0: "batch_size", 1: "time_steps"}, "head_logits": {0: "batch_size", 1: "time_steps"}},
    opset_version=17
)
PYEOF

# 2. Compile the fused ONNX graph into an FP16 TensorRT 10.x engine optimized for NVIDIA L4
trtexec \
  --onnx=model_repository/chordformer_ensemble/1/model.onnx \
  --saveEngine=model_repository/chordformer_ensemble/1/model.plan \
  --fp16 \
  --minShapes=cqt_input:1x431x256 \
  --optShapes=cqt_input:16x431x256 \
  --maxShapes=cqt_input:32x1000x256

# 3. Launch NVIDIA Triton Inference Server with dual-stream dynamic batching
#    (max_batch_size: 32, preferred_batch_size: [8, 16, 32], max_queue_delay_microseconds: 5000, count: 2)
tritonserver --model-repository=/home/shojha/ChordFormer-Artificial-Dataset-Benchmarking/model_repository

# 4. Benchmark concurrency scaling (L=1 to L=16) via Triton perf_analyzer
perf_analyzer -m chordformer_ensemble -u localhost:8001 -i grpc \
  --shape cqt_input:431,256 --concurrency-range 1:16:1
```

---

## Provenance and Attribution

This implementation was developed from code in the upstream repository
[ISMIR2019-Large-Vocabulary-Chord-Recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition),
which accompanied *Large-Vocabulary Chord Transcription via Chord Structure Decomposition* (Jiang et al., ISMIR 2019), and builds upon [ChordFormer: A Conformer-Based Architecture for Large-Vocabulary Audio Chord Recognition](https://ieeexplore.ieee.org/abstract/document/11313568) (Akram et al., 2025/2026). Please cite both works and our preprint when using code, models, or results from this repository.

## Installation

Use Python 3 and install the dependencies listed in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

## Single-File Audio Chord Recognition

Run 5-fold ensemble inference and HMM decoding on a single audio file:

```bash
python3 chord_recognition.py path_to_audio_file path_to_output_file [chord_dict]
# Example:
python3 chord_recognition.py example.mp3 example_chord.lab full
```

## Repository Layout

```text
confor_head16.py                ChordFormer architecture, ReweightedLoss, baseline training entry point
pretrain_synthetic.py           Stage 1 synthetic pretraining (SpecAugment + 12-shift CQT pitch shifting)
finetune_real_fold.py           Stage 2 real-audio 5-fold finetuning (Discriminative LR + Floor Curriculum Replay)
convert_synth_labels.py         Converts synthetic JSON progressions into frame-exact Harte .lab files
create_synth_storage.py         Parallel 252-bin CQT extraction and HDF5 storage builder for synthetic corpus
storage_creation.py             CQT and 6-head label HDF5 cache creation for the 1,217-song real corpus
eval_heads.py                   Fast Numba CRF Viterbi multi-head and per-class evaluation
evaluate_synth_vs_baseline.py   Computes head-wise and per-class recall/F1 deltas vs. control baseline
val_chordformer_head16.py       5-fold XHMM decoding to Harte .lab files across 1,217 real tracks
test_for_all_mirex.py           mir_eval duration-weighted WCSR / MIREX benchmark evaluation
calibrate_biases.py             Out-of-fold post-hoc logit bias calibration and Pareto frontier analysis
model_repository/               Triton Inference Server config and FP16 ONNX/TensorRT ensemble artifacts
extractors/                     Audio, CQT, chord, and XHMMDecoder (xhmm_ismir.py) components
mir/                            Shared MIR neural-network and MixedReplayDataProvider infrastructure
cache_data/                     Pretrained and 5-fold finetuned checkpoints (.sdict)
data/                           Chord vocabularies, 5-fold split CSVs, and class frequency counters
```

## Citation

```bibtex
@article{ojha2026distribution,
  author  = {Ojha, Sharabh},
  title   = {Distribution-Shifted Synthetic Multitrack Pretraining for Large-Vocabulary Audio Chord Recognition},
  journal = {arXiv preprint},
  year    = {2026}
}

@article{akram2025chordformer,
  author  = {Akram, Muhammad Waseem and Dettori, Stefano and Colla, Valentina and Buttazzo, Giorgio Carlo},
  journal = {IEEE Transactions on Audio, Speech and Language Processing},
  title   = {{ChordFormer}: A Conformer-Based Architecture for Large-Vocabulary Audio Chord Recognition},
  year    = {2026},
  volume  = {34},
  pages   = {581--595},
  doi     = {10.1109/TASLPRO.2025.3646468}
}

@inproceedings{jiang2019large,
  author    = {Jiang, Junyan and Chen, Ke and Li, Wei and Xia, Gus},
  title     = {Large-Vocabulary Chord Transcription via Chord Structure Decomposition},
  booktitle = {Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2019},
  pages     = {644--651}
}
```
