#!/usr/bin/env bash
# scripts/compile_tensorrt.sh
# Compiles model_repository/chordformer_trt/1/model.onnx to TensorRT FP16 engine (model.plan)
# Run inside NVIDIA Triton container: nvcr.io/nvidia/tritonserver:24.08-py3

set -euo pipefail

ONNX_PATH="${1:-model_repository/chordformer_trt/1/model.onnx}"
PLAN_PATH="${2:-model_repository/chordformer_trt/1/model.plan}"

echo "=========================================================="
echo "Compiling TensorRT FP16 Engine with Dynamic Batching"
echo "Input ONNX:  ${ONNX_PATH}"
echo "Output Plan: ${PLAN_PATH}"
echo "=========================================================="

trtexec \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${PLAN_PATH}" \
    --fp16 \
    --minShapes=cqt_input:1x431x252 \
    --optShapes=cqt_input:8x431x252 \
    --maxShapes=cqt_input:32x431x252 \
    --workspace=4096 \
    --builderOptimizationLevel=5 \
    --verbose

echo "[SUCCESS] TensorRT plan generated at: ${PLAN_PATH}"
