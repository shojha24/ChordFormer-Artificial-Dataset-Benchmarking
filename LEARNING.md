# ChordFormer ML Systems Design Notes

A systems-level breakdown of the concurrency, throughput, and cost trade-offs across both the **current Triton + TensorRT serving stack** and the **planned raw-audio FastAPI wrapper** documented in [PRODUCTION.md](PRODUCTION.md).

---

## Part 1: Current Serving Stack (GKE + TensorRT + Triton)

The core goal of the serving layer is to move a relatively small ensemble (`30.06 GFLOPs` at `B=1`) out of a memory-bandwidth-bound regime into a compute-bound, pipelined regime on a single NVIDIA L4 GPU (`24 GB` GDDR6, `300 GB/s` bandwidth, `242 TFLOPs` peak FP16).

### 1. Throughput Decisions

* **5-Fold Graph Fusion (`EnsembleChordFormer`) — [`export_ensemble_onnx.py`](export_ensemble_onnx.py)**
  * **Decision:** Fuse all 5 `ChordNet` fold checkpoints, input zero-padding (`252 → 256` for 16-head Conformer alignment), 6-head `softmax` layers, and `1/5` ensemble averaging into one unified ONNX graph.
  * **Why:** Serving 5 separate models behind a Triton Ensemble would incur 5× scheduler hops, 5× intermediate tensor allocations, and 5× host-device handoffs. Fusing them lets TensorRT optimize and execute the entire ensemble in a single pass.

* **On-Target FP16 TensorRT Compilation — [`k8s/trt-compiler-job.yaml`](k8s/trt-compiler-job.yaml)**
  * **Decision:** Compile `model.onnx` → `model.plan` with `trtexec --fp16` directly on the target L4 (`sm_89`) GPU with bounded shapes (`minShapes=1x431x252`, `optShapes=8x431x252`, `maxShapes=32x431x252`).
  * **Why:** FP16 halves weight/activation memory traffic and engages L4 FP16 Tensor Cores, cutting `B=1` compute latency from `36.79 ms` (FP32 PyTorch eager) to `2.83 ms` ([`benchmarks/BENCHMARK_REPORT.md`](benchmarks/BENCHMARK_REPORT.md)). Fixing sequence/feature dimensions around `optShapes=8x431x252` eliminates runtime shape-specialization overhead.

* **Dynamic Batching (`Roofline` Shift) — [`model_repository/chordformer_trt/config.pbtxt`](model_repository/chordformer_trt/config.pbtxt)**
  * **Decision:** Configure `max_batch_size: 32`, `preferred_batch_size: [ 4, 8, 16 ]`, and `max_queue_delay_microseconds: 5000`.
  * **Why:** At `B=1`, streaming ~117.6 MB of ensemble weights across the `300 GB/s` bus for a single segment leaves Tensor Cores underutilized (`26.7%` GPU duty cycle, `92.52 inf/sec`). Allowing up to `5 ms` of queue coalescing (`9.29 ms` measured server queue delay at concurrency 13) amortizes weight loads across `B` parallel segments. Arithmetic intensity scales by $B$, lifting throughput from `92.52` to `395.75 inf/sec` (**4.28× speedup**) while keeping average client round-trip latency at `32.56 ms` (under the `40 ms` SLA).

### 2. Concurrency Decisions

* **Multi-Instance GPU Execution (`count: 2`) — [`model_repository/chordformer_trt/config.pbtxt`](model_repository/chordformer_trt/config.pbtxt)**
  * **Decision:** Run `count: 2` model instances on `gpus: [ 0 ]` (vs. `count: 1`, `max_batch_size: 1` in [`config_baseline.pbtxt`](model_repository/chordformer_trt/config_baseline.pbtxt)).
  * **Why:** Creates two independent CUDA execution streams on the single L4. While Instance 0 runs Tensor Core compute for batch $k$, Instance 1 overlaps PCIe H2D/D2H transfers (`1.27 ms` at peak) and scheduling (`0.82 ms`) for batch $k+1$, supporting up to **64 active segments** (`2 × 32`) in flight simultaneously.

* **gRPC Transport — [`k8s/triton-deployment.yaml`](k8s/triton-deployment.yaml)**
  * **Decision:** Serve inference on port `8001` (gRPC/HTTP2 binary frames) rather than JSON over REST.
  * **Why:** Avoids HTTP/1.1 connection churn and costly JSON serialization/deserialization for `[B, 431, 252]` FP32 inputs (~434 KB per segment).

### 3. Cost Decisions

* **Heterogeneous Split Node Pools — [`PRODUCTION.md`](PRODUCTION.md)**
  * **Decision:** Keep GKE system pods on a lightweight `e2-medium` CPU node (~$25/mo) and schedule only Triton onto a dedicated `g2-standard-4` + NVIDIA L4 node pool (`nvidia.com/gpu: 1`).
  * **Why:** Prevents paying GPU-node rates for cluster system overhead and lets CPU and GPU resources scale independently.

* **Single-GPU Packing + Spot Preemptible Instances (`--spot`)**
  * **Decision:** Maximize utilization of **1 L4 GPU** (~400 seg/sec ≈ 22 full 3-min songs/sec) instead of scaling replicas, and provision `gpu-pool` with `--spot`.
  * **Why:** GPU compute is ~75–85% of the monthly bill. Spot saves ~70% (~$175–$205/mo vs. ~$515–$615/mo On-Demand), trading 30s preemption risk for dramatic cost reduction.

* **Ephemeral Compiler Job + Stateless GCS Model Repository — [`k8s/trt-compiler-job.yaml`](k8s/trt-compiler-job.yaml), [`k8s/triton-deployment.yaml`](k8s/triton-deployment.yaml)**
  * **Decision:** Run `trtexec` inside a one-off Kubernetes `Job` (`trt-compiler`) that deletes `model.onnx` after uploading `model.plan`, and point `tritonserver` directly at `gs://${BUCKET_NAME}/model_repository`.
  * **Why:** Eliminates persistent build VMs, duplicate GCS artifact storage, and expensive `ReadWriteMany` block/Filestore PVCs—allowing `gpu-pool` to scale to `0` when idle without losing state.

---

## Part 2: Planned Raw-Audio API Wrapper (Future Architecture)

Accepting raw audio (`MP3/WAV/FLAC`) introduces CPU-heavy decoding (`ffmpeg`), CQT extraction (`librosa` `CQTV2` in [`extractors/cqt.py`](extractors/cqt.py)), and stateful HMM post-processing (`XHMMDecoder` in [`extractors/xhmm_ismir.py`](extractors/xhmm_ismir.py)). The wrapper design isolates these bottlenecks from the GPU server.

### 1. Throughput & Batching Decisions

* **Intra-Request Song Batching (`[N, 431, 252]` per upload)**
  * **Decision:** Chunk each decoded song into $N$ 10-second windows and send all $N$ chunks in a **single batched gRPC call** to Triton (`chordformer_trt`).
  * **Why:** A 3-minute song produces ~18 chunks, fitting inside `max_batch_size: 32`. Even at **1 concurrent user**, the GPU immediately executes at `B=18` in its compute-bound sweet spot rather than issuing 18 sequential `B=1` calls.

* **Startup Singleton Caching (`XHMMDecoder`)**
  * **Decision:** Initialize `XHMMDecoder` once per worker at FastAPI startup rather than per request.
  * **Why:** Avoids repeatedly reading `data/*_chord_list.txt` and rebuilding the chord template/transition state space on every upload.

### 2. Concurrency & Stability Decisions

* **Decoupled CPU Preprocessing & GIL Isolation (`ProcessPoolExecutor`)**
  * **Decision:** Keep audio decoding, `CQTV2`, and `XHMMDecoder` inside FastAPI CPU pods and run `CQTV2` inside a bounded `ProcessPoolExecutor`.
  * **Why:** Keeps GPU memory/compute free of Python GIL contention and audio I/O, while preventing CPU-bound `librosa`/NumPy calls from blocking FastAPI's async event loop (which would stall `/healthz`, `/readyz`, and gRPC I/O under load). Bounding the process pool keeps pod RAM within `1–2 GiB`.

* **Asymmetric Autoscaling (CPU HPA + Cluster Autoscaler vs. Fixed 1× GPU)**
  * **Decision:** Upgrade the CPU pool (`e2-medium` → `e2-standard-4`) and attach a `HorizontalPodAutoscaler` (`minReplicas: 1`, `maxReplicas: 8`, 70% CPU target) + GKE Cluster Autoscaler, while keeping Triton at 1 L4 pod.
  * **Why:** On the initial `e2-medium` (~1–1.5 allocatable vCPUs), CPU CQT bottlenecks raw-audio concurrency to **~5–15 concurrent requests** long before the L4 GPU (~22 songs/sec) is saturated. Scaling CPU pods horizontally unlocks the GPU's full capacity without paying for a second GPU.

* **Upload Guardrails & Timeout Alignment**
  * **Decision:** Enforce max upload size (e.g., `25 MB`) and audio duration caps at the API boundary, and configure Ingress timeouts to `30–60s`.
  * **Why:** Prevents malicious or oversized uploads from triggering OOM kills during PCM resampling/CQT allocation, and avoids premature `504` gateway timeouts on multi-minute tracks.

* **Phase 3 Escape Hatch: GPU CQT Offloading (`nnAudio` Triton Ensemble)**
  * **Decision:** Defer GPU-accelerated CQT (`nnAudio.features.CQT` in a Triton Python/Ensemble backend) until raw-audio traffic makes CPU nodes more expensive than the GPU.
  * **Why:** Keeps initial implementation simple with exact parity to `extractors/cqt.py`, while preserving a clean architectural path to trade spare L4 GPU cycles for zero CPU CQT scaling cost at high volume.

### 3. Cost & Availability Strategy

* **Internal `ClusterIP` for Triton + Single External Load Balancer**
  * **Decision:** Switch `triton-chordformer-service` from `LoadBalancer` to `ClusterIP` once the API wrapper is deployed, exposing only the API Ingress/LoadBalancer publicly.
  * **Why:** Eliminates a redundant ~$18/mo GCP Load Balancer and removes direct public exposure of the Triton gRPC/HTTP ports.

* **Tiered Cost / HA Progression**
  * **Tier 1 — MVP / Dev (`~$240–$270/mo`):** Validate end-to-end on the existing `1 × e2-medium` + `1 × Spot L4` (~5–15 concurrent uploads, $0 extra infra).
  * **Tier 2 — Prod Balanced (`~$315–$350/mo`):** Swap `e2-medium` for `e2-standard-4` + HPA (~20–40+ concurrent uploads, autoscalable).
  * **Tier 3 — Prod HA (`~$655–$870/mo`):** Remove `--spot` to eliminate preemption risk (~40–80+ concurrent uploads).
  * **Cost Levers:** For 24/7 HA, a **1-year Committed Use Discount (CUD)** cuts On-Demand `g2-standard-4` GPU cost by ~37% (~$330–$390/mo). For dev/intermittent workloads, scaling `gpu-pool` to `0` when idle drops GPU spend to **~$0.25/hr** only while active (~$30/mo at 4 hrs/day).
