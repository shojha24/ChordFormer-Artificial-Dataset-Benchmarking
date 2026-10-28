#!/usr/bin/env bash
# scripts/run_benchmarks.sh
# Runs NVIDIA perf_analyzer against the deployed Triton endpoint to generate
# baseline.csv and tuned_dynamic_batching.csv for the resume metrics.

set -euo pipefail

TRITON_URL="${1:-localhost:8001}"
MODEL_NAME="chordformer_trt"
OUTPUT_DIR="${2:-benchmarks}"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================================="
echo "ACR Triton & TensorRT Benchmark Verification Suite"
echo "Triton Target: ${TRITON_URL}"
echo "Model Name:    ${MODEL_NAME}"
echo "Output Dir:    ${OUTPUT_DIR}"
echo "=========================================================="

echo "[1/2] Running Baseline Benchmark (Unbatched, Concurrency = 1)..."
perf_analyzer \
    -m "${MODEL_NAME}" \
    -u "${TRITON_URL}" \
    -i grpc \
    --shape cqt_input:431,252 \
    --concurrency-range 1:1 \
    --measurement-interval 10000 \
    -f "${OUTPUT_DIR}/baseline.csv"

echo "[SUCCESS] Baseline benchmark saved to ${OUTPUT_DIR}/baseline.csv"

echo "[2/2] Running Tuned Dynamic Batching Benchmark (Concurrency Sweep 1 to 32)..."
perf_analyzer \
    -m "${MODEL_NAME}" \
    -u "${TRITON_URL}" \
    -i grpc \
    --shape cqt_input:431,252 \
    --concurrency-range 1:32:4 \
    --latency-threshold 50 \
    --measurement-interval 10000 \
    -f "${OUTPUT_DIR}/tuned_dynamic_batching.csv"

echo "[SUCCESS] Tuned dynamic batching benchmark saved to ${OUTPUT_DIR}/tuned_dynamic_batching.csv"
echo ""
echo "Verification checklist:"
echo "1. Verify 'compute infer' latency remains < 40,000 usec (40ms)."
echo "2. Check throughput ratio (max tuned throughput / baseline throughput >= 4.2x)."
