# ChordFormer Production Deployment Guide (GKE + TensorRT + Triton)

This reference lists the minimal, ordered commands required to compile the ChordFormer ONNX model into an FP16 TensorRT engine (`model.plan`) on an NVIDIA L4 GPU and serve it on Google Kubernetes Engine (GKE) with Triton Inference Server.

---

## Prerequisites

- Authenticated `gcloud` CLI with a default project configured (`gcloud auth login`, `gcloud config set project <PROJECT_ID>`).
- `kubectl` and `envsubst` (from `gettext`) installed locally.
- Local `model_repository/` populated with:
  - `model_repository/chordformer_trt/1/model.onnx`
  - `model_repository/chordformer_trt/config.pbtxt`

---

## 1. Environment Variables & GCS Setup

Set environment variables, enable required GCP APIs, create the model repository bucket, grant the Compute Engine default service account GCS access, and upload the local `model_repository/`.

```bash
# Set GCP project, region, zone, and bucket variables
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export ZONE="us-central1-a"
export BUCKET_NAME="${PROJECT_ID}-chordformer-triton"
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")

# Enable Compute Engine and GKE APIs
gcloud services enable container.googleapis.com compute.googleapis.com

# Create GCS bucket for the Triton model repository
gcloud storage buckets create gs://${BUCKET_NAME} --location=${REGION}

# Grant GKE nodes read/write/delete access to the bucket (needed for TensorRT compilation & Triton loading)
gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Upload model_repository (model.onnx + config.pbtxt) to GCS
gcloud storage cp -r model_repository gs://${BUCKET_NAME}/
```

---

## 2. Provision GKE Cluster & NVIDIA L4 GPU Node Pool

Create a lightweight CPU control-plane node pool and attach an NVIDIA L4 GPU node pool.

> **Production Note:** The `--spot` flag reduces GPU cost by ~70% but allows GCP to preempt the node. Remove `--spot` for strict high-availability production deployments.

```bash
# 1. Create the GKE cluster (also configures kubectl credentials automatically)
gcloud container clusters create acr-triton-cluster \
    --zone=${ZONE} \
    --num-nodes=1 \
    --machine-type=e2-medium \
    --scopes=https://www.googleapis.com/auth/devstorage.full_control

# 2. Add the NVIDIA L4 GPU node pool
gcloud container node-pools create gpu-pool \
    --cluster=acr-triton-cluster \
    --zone=${ZONE} \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1,gpu-driver-version=latest \
    --spot \
    --num-nodes=1 \
    --scopes=https://www.googleapis.com/auth/devstorage.full_control
```

*(Only needed if reconnecting from a new machine/session:)*
```bash
gcloud container clusters get-credentials acr-triton-cluster --zone=${ZONE}
```

---

## 3. Compile FP16 TensorRT Engine on L4 GPU

TensorRT engines are hardware-specific and must be compiled on the target L4 GPU. The `k8s/trt-compiler-job.yaml` Job downloads `model.onnx` from GCS, compiles `model.plan` with `trtexec --fp16`, uploads `model.plan` back to GCS, and deletes `model.onnx` from GCS.

```bash
# Launch the TensorRT compilation Job
envsubst '${BUCKET_NAME}' < k8s/trt-compiler-job.yaml | kubectl apply -f -

# Wait for the container image to pull and start (~45-60s on a fresh node)
kubectl wait --for=condition=Ready pod -l app=trt-compiler --timeout=300s

# Stream live trtexec compilation logs until completion
kubectl logs -f job/trt-compiler

# Clean up the completed compilation Job
kubectl delete job trt-compiler
```

*(Optional — Least Privilege Hardening):* Once `model.plan` is uploaded and compilation is finished, Triton only needs read access to GCS. You can downgrade the node service account from `objectAdmin` to `objectViewer`:

```bash
gcloud storage buckets remove-iam-policy-binding gs://${BUCKET_NAME} \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
```

---

## 4. Deploy Triton Inference Server

Deploy the Triton `Deployment` and `LoadBalancer` `Service` (`k8s/triton-deployment.yaml`), which loads `model.plan` directly from `gs://${BUCKET_NAME}/model_repository`.

```bash
# Deploy Triton server and LoadBalancer service
envsubst '${BUCKET_NAME}' < k8s/triton-deployment.yaml | kubectl apply -f -

# Wait for the deployment rollout to become ready (~45s to pull image & load model.plan)
kubectl rollout status deployment/triton-chordformer

# Check external LoadBalancer IP and ports (8000 HTTP, 8001 gRPC, 8002 Metrics)
kubectl get svc triton-chordformer-service
```

Verify the server is healthy once an `EXTERNAL-IP` is assigned:
```bash
curl -i http://<EXTERNAL-IP>:8000/v2/health/ready
```

---

## 5. Raw-Audio API Wrapper: Implementation Plan & Cost Analysis

To accept raw audio uploads (MP3/WAV/FLAC) instead of precomputed `[431, 252]` CQT tensors, deploy a FastAPI wrapper in front of Triton on the CPU node pool.

### Architecture & Request Flow

```text
Client (MP3/WAV/FLAC upload)
   │  HTTPS
   ▼
GKE Ingress / LoadBalancer (TLS termination)
   │
   ▼
FastAPI Wrapper Pods (CPU node pool, HPA autoscaled)
   ├── 1. Decode & resample audio → 22,050 Hz mono (ffmpeg / soundfile)
   ├── 2. Extract CQTV2 → slice to 252 bins, chunk into [N, 431, 252] (10s windows)
   ├── 3. Batched gRPC call → triton-chordformer-service:8001 (internal ClusterIP)
   ├── 4. Stitch 6 probability heads & run XHMMDecoder (extractors/xhmm_ismir.py)
   └── 5. Return JSON [{start, end, chord}, ...] (or .lab)
   │
   ▼
Triton Pod (existing g2-standard-4 + L4 GPU)
```

> **Security & Cost Tip:** Once the API wrapper is deployed, change `triton-chordformer-service` in `k8s/triton-deployment.yaml` from `type: LoadBalancer` to `type: ClusterIP` so Triton is only exposed inside the cluster and you only pay for **one** external Load Balancer.

### Capacity Reference (Existing Single L4 GPU Pod)

Based on `benchmarks/BENCHMARK_REPORT.md` (`config.pbtxt`: `count: 2` GPU instances, `max_batch_size: 32`):
- **Triton/L4 GPU Ceiling:** ~395–450 10s-segments/sec (~22 full 3-min songs/sec), with 32.6 ms avg / 10.93 ms compute latency at concurrency 13.
- **CPU Bottleneck Warning:** On the initial `1 × e2-medium` CPU node (~1–1.5 allocatable vCPUs after GKE system pods), CPU-bound `librosa` CQT extraction limits raw-audio concurrency to **~5–15 concurrent requests** before the CPU node saturates. Scaling the CPU pool (`e2-standard-4` + HPA) unlocks the GPU's full capacity.

### Implementation Plan

#### Phase 1: Build the FastAPI Service (`api/`)
1. **Create `api/main.py`:**
   - `POST /v1/predict` accepting `multipart/form-data` audio (`UploadFile`, enforce max file size e.g. 25 MB and duration cap) + optional `chord_dict` (`submission`, `ismir2017`, `full`, `extended`).
   - `GET /healthz` (liveness) and `GET /readyz` (verifies gRPC health against Triton `/v2/health/ready`).
2. **Preprocess, Batch, & Decode (`api/pipeline.py`):**
   - Decode audio via `soundfile` / `ffmpeg` to mono `22,050 Hz` (`settings.py`: `DEFAULT_SR=22050`, `DEFAULT_HOP_LENGTH=512`).
   - Run `CQTV2` (`extractors/cqt.py`, sliced to 252 bins) inside a bounded `ProcessPoolExecutor` so CPU-bound `librosa` work does not block FastAPI's async event loop.
   - Pad/chunk frames into `[N, 431, 252]` FP32 batches (e.g., a 3-min song = 18 chunks, which fits in a single Triton batch since `max_batch_size: 32`).
   - Send over `tritonclient.grpc.aio` to `triton-chordformer-service:8001` (`chordformer_trt`).
   - Trim padded frames and decode the 6 probability heads (`root_triad`, `bass`, `seventh`, `ninth`, `eleventh`, `thirteenth`) using a pre-loaded singleton `XHMMDecoder` (`extractors/xhmm_ismir.py`) initialized once at startup.
3. **Containerize (`api/Dockerfile`):**
   - Base image `python:3.10-slim` + `ffmpeg` + `libsndfile1` + `fastapi`, `uvicorn`, `tritonclient[grpc]`, `librosa`, `numpy`, `scipy`.
   - Push image to GCP Artifact Registry (`us-central1-docker.pkg.dev/${PROJECT_ID}/chordformer/api:latest`).

#### Phase 2: Kubernetes Deployment & Autoscaling (`k8s/`)
1. **Update `k8s/triton-deployment.yaml`:**
   - Switch `triton-chordformer-service` from `LoadBalancer` to `ClusterIP`.
2. **Add `k8s/api-deployment.yaml`:**
   - `Deployment` for the FastAPI service scheduled on the CPU pool (`requests: {cpu: "1", memory: "1Gi"}`, `limits: {cpu: "2", memory: "2Gi"}`).
   - `Service` (`ClusterIP`) + GKE `Ingress` (or `LoadBalancer`) with HTTPS/TLS termination and 30–60s request timeout.
3. **Add `k8s/api-hpa.yaml` (`HorizontalPodAutoscaler`):**
   - Scale API pods on CPU utilization (target 70%), e.g., `minReplicas: 1`, `maxReplicas: 8`.
   - Enable GKE Cluster Autoscaler on the CPU node pool so extra CPU nodes spin up automatically during traffic bursts.

#### Phase 3: Optional GPU Preprocessing Optimization
- If raw-audio traffic grows such that CPU CQT extraction becomes the dominant cost, move CQT extraction onto the existing L4 GPU using `nnAudio.features.CQT` inside a Triton Python/Ensemble backend, leaving only lightweight audio decoding and HMM decoding on CPU pods.

### Monthly Cost Analysis (`us-central1`, 24/7 Estimates)

*Estimates based on standard GCP `us-central1` list pricing (730 hrs/month), including the existing Triton GPU pod.*

| Component | Tier 1: MVP / Dev (Current Nodes, Spot GPU) | Tier 2: Prod Balanced (Spot GPU + Scaled CPU Pool) | Tier 3: Prod HA (On-Demand GPU + Scaled CPU Pool) |
| :--- | :---: | :---: | :---: |
| **GKE Zonal Control Plane** | $0* *(or $73)* | $0* *(or $73)* | $0* *(or $73)* |
| **Existing GPU Node** (`1 × g2-standard-4` + L4) | ~$175–$205 *(Spot)* | ~$175–$205 *(Spot)* | ~$515–$615 *(On-Demand)* |
| **CPU Node(s) for API + System** | ~$25 (`1 × e2-medium`) | ~$98 (`1 × e2-standard-4`, replaces `e2-medium`) | ~$98–$196 (`1–2 × e2-standard-4`) |
| **Boot Disks** (`pd-balanced`, 100 GB/node) | ~$20 (2 nodes) | ~$20 (2 nodes) | ~$20–$30 (2–3 nodes) |
| **External Load Balancer / Ingress** | ~$18 + traffic | ~$18 + traffic | ~$18 + traffic |
| **GCS + Artifact Registry + Egress** | <$2 | ~$2–$5 | ~$5–$10 |
| **Estimated Monthly Total (24/7)** | **~$240–$270/mo** | **~$315–$350/mo** | **~$655–$870/mo** |
| **Est. Concurrent Raw-Audio Capacity** | **~5–15 concurrent** | **~20–40 concurrent** *(autoscalable)* | **~40–80+ concurrent** *(HA, no preemption)* |

*\*GCP provides a $74.40/month free credit covering **one** zonal GKE cluster control plane per billing account. If you run additional clusters, add **$73/month**.*

#### Cost Optimization Levers
- **Idle GPU scale-to-zero:** For dev/non-24/7 workloads, scaling `gpu-pool` to `0` nodes when idle reduces GPU cost to **~$0.25/hr only while running** (e.g., 4 hrs/day ≈ **$30/mo**).
- **Spot vs. 1-Year CUD:** Spot saves ~$350–$400/mo on `g2-standard-4` (with 30s preemption risk); for 24/7 HA without preemption, a 1-year Committed Use Discount cuts On-Demand GPU cost by ~37% (~$330–$390/mo).
- **Start on Tier 1, scale to Tier 2:** Deploy the API onto the existing `e2-medium` first to validate end-to-end with $0 added compute cost, then upgrade the CPU pool to `e2-standard-4` + HPA once concurrent upload demand exceeds ~10 requests.

---

## Optional: Performance Benchmarking (`perf_analyzer`)

Not required to serve the model, but useful for verifying latency and dynamic-batching throughput from inside the cluster.

```bash
# 1. Start the Triton SDK client pod and wait until it is Ready
kubectl run perf-client --restart=Never \
    --image=nvcr.io/nvidia/tritonserver:24.08-py3-sdk \
    -- sleep 3600
kubectl wait --for=condition=Ready pod/perf-client --timeout=180s

# 2. Run single-concurrency baseline and dynamic-batching concurrency sweep
kubectl exec -it perf-client -- perf_analyzer \
    -m chordformer_trt \
    -u triton-chordformer-service:8001 \
    -i grpc \
    --shape cqt_input:431,252 \
    --concurrency-range 1:1 \
    -f /tmp/baseline.csv

kubectl exec -it perf-client -- perf_analyzer \
    -m chordformer_trt \
    -u triton-chordformer-service:8001 \
    -i grpc \
    --shape cqt_input:431,252 \
    --concurrency-range 1:32:4 \
    -f /tmp/tuned_dynamic_batching.csv

# 3. Copy benchmark CSVs locally and inspect results
kubectl cp perf-client:/tmp/baseline.csv baseline.csv
kubectl cp perf-client:/tmp/tuned_dynamic_batching.csv tuned_dynamic_batching.csv
cat tuned_dynamic_batching.csv

# 4. Clean up the benchmark pod
kubectl delete pod perf-client
```

---

## Teardown / Cost Cleanup

Run this when you want to shut down the cluster and stop GPU billing.

```bash
# Delete the GKE cluster (stops all compute/GPU charges)
gcloud container clusters delete acr-triton-cluster --zone=${ZONE} --quiet

# Optional: delete the GCS bucket and stored model artifacts
# gcloud storage rm -r gs://${BUCKET_NAME}
```
