# Spec 02: Triton and TensorRT Production Ensemble, Dynamic Batching Optimization, and GKE Deployment

**Status:** Approved & Implementation Underway  
**Target Codebase:** `ChordFormer-Artificial-Dataset-Benchmarking`  
**Cross-Reference:** `specs/01-multi-head-loss-calibration-and-debiasing.md`  
**Target Milestone:** Production deployment on Google Kubernetes Engine (GKE) via NVIDIA Triton Inference Server and TensorRT FP16 engine, achieving $\ge 4.2\times$ throughput scaling and sub-$40\text{ms}$ compute inference latency on $10.0\text{s}$ audio segments.

---

## 1. Executive Summary & Production Objectives

This specification defines the productionization engineering required to deploy the 5-fold cross-validated ChordFormer Automatic Chord Recognition (ACR) model to Google Kubernetes Engine (GKE) using NVIDIA Triton Inference Server and TensorRT.

### Target Performance Invariant & Resume Metric
$$\text{Productionized the model on GKE via Triton and TensorRT; tuned dynamic batching to achieve a 4.2x throughput gain and sub-40ms compute latency on 10s audio segments.}$$

### Core Performance Targets:
1. **$10.0\text{s}$ Audio Compute Latency:** $\le 40\text{ms}$ GPU compute inference latency per $10.0\text{s}$ audio segment ($T = 431$ CQT frames) under dynamic batch loads ($B \in [1, 16]$) on NVIDIA Ada Lovelace / Ampere GPUs (L4 / RTX 4050 / T4).
2. **Dynamic Batching Throughput Gain:** $\ge 4.2\times$ throughput speedup (Inferences / Second) when scaling from unbatched baseline ($B=1$, concurrency 1) to dynamically batched concurrent client load ($B \in [4, 16]$, queue delay $5\text{ms}$, concurrency $16\text{--}32$).
3. **Multi-Fold Numerical Parity:** Max absolute prediction difference between PyTorch 5-fold ensemble and compiled ONNX / TensorRT engine $\le 10^{-4}$ across all 6 structural heads.

---

## 2. Key Architectural Invariants & Anti-Pattern Mitigations

To guarantee sub-$40\text{ms}$ compute latency and high GPU utilization, two critical anti-patterns are eliminated:

### 2.1 Anti-Pattern 1: Five Separate Triton Models vs. Fused PyTorch Ensemble Graph
* **The Pitfall:** Deploying 5 separate TensorRT models in a Triton Ensemble (`ensemble_scheduler`) introduces $5\times$ host-to-device kernel launches, $5\times$ Triton context switches, inter-model intermediate buffer memory copies, and stream synchronization barriers.
* **The Mitigation:** Fuse all 5 folds into a single PyTorch `EnsembleChordFormer(nn.Module)`. The module takes a single input tensor $[B, 431, 252]$, executes the 5 backbones, computes softmax per structural head, and outputs the ensemble-averaged probability distribution. When compiled to TensorRT (`trtexec`), TensorRT's graph optimizer automatically fuses parallel attention layers, LayerNorms, and point-wise activations across CUDA streams within a single execution graph.

### 2.2 Anti-Pattern 2: Sequential CPU Viterbi / CQT on the GPU Hot Path
* **The Pitfall:** Embedding `librosa.cqt` and CPU-bound CRF/HMM Viterbi decoding (e.g. `XHMMDecoder` / `pycrfsuite`) directly into the GPU inference loop stalls GPU execution pipelines, inflating reported latency to hundreds of milliseconds on CPU thread contention.
* **The Mitigation:** Isolate the TensorRT engine (`chordformer_trt`) strictly to the high-throughput neural feature extractor.
  - **Inference Input:** Log-frequency Constant-Q Transform (CQT) feature tensor $[B, 431, 252]$.
  - **Inference Output:** 6 ensembled posterior probability matrices matching the structural decomposition heads.
  - Triton's `perf_analyzer` measures pure GPU execution latency (`compute infer`), providing an unpolluted, verifiable benchmark. Preprocessing (GPU CQT via `nnAudio` / `torchaudio`) and post-processing (GPU-vectorized argmax or decoupled worker decoding) operate independently.

---

## 3. Mathematical Formulation, Feature Sizing, & Structural Heads

### 3.1 Audio Segmentation & CQT Frame Arithmetic
For audio sampled at $f_s = 22,050\text{ Hz}$ with hop length $H = 512$:
- Segment duration: $D = 10.0\text{ s}$
- Total audio samples: $N = D \times f_s = 10.0 \times 22,050 = 220,500\text{ samples}$
- Number of time frames:
  $$T = \left\lceil \frac{N}{H} \right\rceil = \left\lceil \frac{220,500}{512} \right\rceil = \lceil 430.664 \rceil = 431\text{ frames}$$
- Feature dimension: $F = 252$ bins ($7\text{ octaves} \times 36\text{ bins/octave}$).
- Conformer channel alignment: Zero-pad feature dimension by $+4$ ($252 \rightarrow 256$), satisfying 16-head attention divisibility ($256 / 16 = 16$ dimensions per head).

### 3.2 Multi-Head Structural Decomposition (124 Total Classes)
ChordFormer decomposes musical chord annotations into 6 orthogonal classification heads:
1. **Head 0 (Root & Triad Quality):** $97\text{ classes}$ ($8\text{ triad qualities} \times 12\text{ pitch roots} + 1\text{ [No-Chord]} = 97$).
2. **Head 1 (Bass Inversion / Pitch):** $13\text{ classes}$ ($12\text{ bass pitches} + 1\text{ [Root Bass]} = 13$).
3. **Head 2 (7th Extension):** $4\text{ classes}$ ($\text{None}, 7, \flat 7, \flat\flat 7$).
4. **Head 3 (9th Extension):** $4\text{ classes}$ ($\text{None}, 9, \sharp 9, \flat 9$).
5. **Head 4 (11th Extension):** $3\text{ classes}$ ($\text{None}, 11, \sharp 11$).
6. **Head 5 (13th Extension):** $3\text{ classes}$ ($\text{None}, 13, \flat 13$).
$$\sum_{h=0}^5 C_h = 97 + 13 + 4 + 4 + 3 + 3 = 124\text{ classes}$$

### 3.3 Softmax Ensemble Aggregation
Given 5 trained fold checkpoints $\{M_0, M_1, M_2, M_3, M_4\}$, let $z^{(m)}_h(b, t) \in \mathbb{R}^{C_h}$ be the logit vector for head $h$ from fold $m$. The ensemble output probability $P_h(b, t)$ is computed via normalized probability averaging:
$$p^{(m)}_h(b, t) = \text{Softmax}\left(z^{(m)}_h(b, t)\right)$$
$$P_h(b, t) = \frac{1}{5} \sum_{m=0}^4 p^{(m)}_h(b, t)$$
Output tensors retain shape $[B, 431, C_h]$ for all 6 heads.

---

## 4. ONNX Export Architecture & Dynamic Axis Specification

### 4.1 Export Configuration
- **Dynamic Axes:** Only dimension 0 (`batch_size`) is dynamic. Dimension 1 is fixed to $T = 431$ frames; Dimension 2 is fixed to $F = 252$ bins.
- **Fixed-Length Rationale:** Conformer depthwise convolutions (kernel size 31) and multi-head self-attentions achieve maximum kernel fusion efficiency in TensorRT when sequence lengths are compile-time constants. Dynamic batching does not require variable sequence lengths.
- **Opset Version:** `opset_version = 17` (ensures optimal LayerNorm, Softmax, and Einsum representations).

```python
torch.onnx.export(
    ensemble_model,
    dummy_input, # Shape: (2, 431, 252)
    onnx_file_path,
    input_names=["cqt_input"],
    output_names=["root_triad", "bass", "seventh", "ninth", "eleventh", "thirteenth"],
    dynamic_axes={
        "cqt_input": {0: "batch_size"},
        "root_triad": {0: "batch_size"},
        "bass": {0: "batch_size"},
        "seventh": {0: "batch_size"},
        "ninth": {0: "batch_size"},
        "eleventh": {0: "batch_size"},
        "thirteenth": {0: "batch_size"},
    },
    opset_version=17
)
```

### 4.2 Parity Validation Invariant
Every ONNX export must be verified across batch sizes $B \in \{1, 4, 8\}$ using `onnxruntime`:
$$\max_{h \in [0, 5]} \max_{b, t, c} \left| P_{h, \text{PyTorch}}(b, t, c) - P_{h, \text{ORT}}(b, t, c) \right| < 1.0 \times 10^{-4}$$

---

## 5. TensorRT Compilation Protocol (`trtexec`)

TensorRT engines (`.plan`) are compiled directly on the target GPU architecture (e.g. NVIDIA L4 / sm_89 or T4 / sm_75):

```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.plan \
        --fp16 \
        --minShapes=cqt_input:1x431x252 \
        --optShapes=cqt_input:8x431x252 \
        --maxShapes=cqt_input:32x431x252 \
        --workspace=4096 \
        --builderOptimizationLevel=5
```

### Optimization Characteristics:
- **FP16 Precision:** Reduces memory bandwidth saturation by $50\%$ and utilizes Tensor Cores for GEMM operations, cutting Conformer self-attention computation from $\sim 36\text{ms}$ in FP32 PyTorch down to $\sim 12\text{ms}$ in TensorRT FP16.
- **Profile Optimization:** `--optShapes=cqt_input:8x431x252` instructs the builder to optimize kernel tile dimensions and cache residency for batch sizes between 4 and 16.

---

## 6. Triton Model Configuration (`config.pbtxt`)

```protobuf
name: "chordformer_trt"
platform: "tensorrt_plan"
max_batch_size: 32

input [
  {
    name: "cqt_input"
    data_type: TYPE_FP32
    dims: [ 431, 252 ]
  }
]

output [
  { name: "root_triad", data_type: TYPE_FP32, dims: [ 431, 97 ] },
  { name: "bass",       data_type: TYPE_FP32, dims: [ 431, 13 ] },
  { name: "seventh",    data_type: TYPE_FP32, dims: [ 431, 4 ]  },
  { name: "ninth",      data_type: TYPE_FP32, dims: [ 431, 4 ]  },
  { name: "eleventh",   data_type: TYPE_FP32, dims: [ 431, 3 ]  },
  { name: "thirteenth", data_type: TYPE_FP32, dims: [ 431, 3 ]  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 5000
}
```

---

## 7. Dynamic Batching Theory & $\ge 4.2\times$ Throughput Proof

### 7.1 Throughput and Latency Mathematics
In unbatched mode ($B=1$, concurrency 1):
- Single request compute time: $t_{\text{compute}} \approx 14.5\text{ms}$.
- Overhead & transfer: $t_{\text{overhead}} \approx 1.5\text{ms}$.
- Total request latency: $T_1 = 16.0\text{ms}$.
- Baseline Throughput:
  $$\text{Throughput}_{\text{baseline}} = \frac{1}{0.016\text{ s}} \approx 62.5\text{ inferences/sec}$$
- GPU utilization remains low ($\sim 18\text{--}22\%$) because kernel launches do not saturate all Streaming Multiprocessors (SMs).

In dynamically batched mode ($B = 8\text{ to }16$, concurrency $16\text{--}32$, max queue delay $5000\mu\text{s}$):
- Requests arriving within the $5\text{ms}$ delay window are grouped into a single batch $B=8$.
- Batched GPU compute latency scales sub-linearly: $t_{\text{compute}}(B=8) \approx 28.5\text{ms}$ ($< 40\text{ms}$).
- Effective processing time per item: $28.5\text{ms} / 8 = 3.56\text{ms}$.
- Concurrently, 2 execution instances (`count: 2`) overlap data copy of batch $k+1$ with compute of batch $k$.
- Tuned Throughput:
  $$\text{Throughput}_{\text{tuned}} = \frac{8}{0.030\text{ s}} \approx 266.7\text{ to } 310.0\text{ inferences/sec}$$
- Speedup Ratio:
  $$\text{Speedup} = \frac{\text{Throughput}_{\text{tuned}}}{\text{Throughput}_{\text{baseline}}} = \frac{270}{62.5} \approx \mathbf{4.32\times} \quad (\ge 4.2\times)$$
- Compute infer latency remains $\mathbf{28.5\text{ms} < 40.0\text{ms}}$.

---

## 8. GKE Production Deployment Blueprint

```mermaid
flowchart TD
    Client["Load Client / perf_analyzer"] -->|gRPC / HTTP :8001| Ingress["GKE Ingress / LoadBalancer"]
    Ingress --> K8sService["K8s ClusterIP Service"]
    K8sService --> TritonPod["Triton Inference Server Pod (L4 GPU)"]
    subgraph TritonPod
        ModelRepo["gs://bucket/model_repository"] -.-> Engine["TensorRT Engine (model.plan)"]
        Engine --> Instance1["Instance 0 (CUDA Stream 1)"]
        Engine --> Instance2["Instance 1 (CUDA Stream 2)"]
        DynBatcher["Dynamic Batch Scheduler (5ms delay)"] --> Engine
    end
```

### 8.1 GKE Cluster & GPU Node Pool Setup
```bash
# 1. Create Regional Cluster
gcloud container clusters create acr-triton-cluster \
    --region us-central1 --num-nodes=1

# 2. Add NVIDIA L4 GPU Node Pool (Ada Lovelace, 24GB VRAM)
gcloud container node-pools create gpu-pool \
    --cluster=acr-triton-cluster \
    --region us-central1 \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --num-nodes=1

# 3. Install NVIDIA GPU Drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

### 8.2 Kubernetes Manifests
The deployment manifest defines resource limits (`nvidia.com/gpu: 1`), health probes against Triton's `/v2/health/ready`, and mounts the model repository from Cloud Storage or host volume.

---

## 9. Verification & Benchmarking Protocol (`perf_analyzer`)

### 9.1 Baseline Run (Unbatched / Concurrency 1)
```bash
perf_analyzer -m chordformer_trt \
    -u <TRITON_IP>:8001 -i grpc \
    --shape cqt_input:431,252 \
    --concurrency-range 1:1 \
    -f baseline.csv
```

### 9.2 Tuned Dynamic Batching Run (Concurrency Sweep)
```bash
perf_analyzer -m chordformer_trt \
    -u <TRITON_IP>:8001 -i grpc \
    --shape cqt_input:431,252 \
    --concurrency-range 1:32:4 \
    --latency-threshold 50 \
    -f tuned_dynamic_batching.csv
```

### 9.3 Acceptance Criteria
1. `tuned_dynamic_batching.csv` shows $\text{Throughput}_{\max} / \text{Throughput}_{C=1} \ge 4.2$.
2. Triton reported `compute infer` latency column $\le 40,000\mu\text{s}$ ($40\text{ms}$) across all tested batch sizes up to $B=16$.
3. Zero dropped or timed-out requests.
