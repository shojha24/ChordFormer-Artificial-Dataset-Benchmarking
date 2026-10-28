"""
test_ensemble_onnx.py

Unit test suite for validating:
1. EnsembleChordFormer PyTorch forward pass output shape and probability constraints.
2. ONNX graph structure, dynamic batching shape inference.
3. Numerical output parity between PyTorch and ONNX Runtime (error <= 1e-4).
"""

import os
import sys
import numpy as np
import pytest
import torch
import onnxruntime as ort

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from export_ensemble_onnx import EnsembleChordFormer

ONNX_MODEL_PATH = os.path.join(REPO_ROOT, "model_repository/chordformer_trt/1/model.onnx")
CHECKPOINTS = [
    os.path.join(REPO_ROOT, f"cache_data/chordformer_head16_synth_ft_loss_s{i}.best.sdict")
    for i in range(5)
]


@pytest.fixture(scope="module")
def ensemble_model():
    model = EnsembleChordFormer(CHECKPOINTS, device="cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def ort_session():
    assert os.path.exists(ONNX_MODEL_PATH), f"ONNX model file not found at {ONNX_MODEL_PATH}"
    sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    return sess


def test_pytorch_ensemble_forward(ensemble_model):
    batch_size = 2
    dummy_input = torch.randn(batch_size, 431, 252)
    with torch.no_grad():
        outputs = ensemble_model(dummy_input)

    assert len(outputs) == 6, "Expected 6 structural heads"
    head_dims = [97, 13, 4, 4, 3, 3]

    for i, (out, expected_dim) in enumerate(zip(outputs, head_dims)):
        assert out.shape == (batch_size, 431, expected_dim), f"Head {i} shape mismatch: {out.shape}"
        # Probability validity: values in [0, 1] and sum along last dim ~ 1.0
        sum_probs = out.sum(dim=-1)
        assert torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-4), f"Head {i} probabilities do not sum to 1.0"
        assert (out >= 0.0).all() and (out <= 1.0).all(), f"Head {i} contains invalid probabilities"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_onnx_runtime_parity(ensemble_model, ort_session, batch_size):
    x_torch = torch.randn(batch_size, 431, 252)
    with torch.no_grad():
        pt_outs = ensemble_model(x_torch)

    ort_outs = ort_session.run(None, {"cqt_input": x_torch.numpy()})
    assert len(ort_outs) == 6, "ORT outputs count mismatch"

    for i, (pt_out, ort_out) in enumerate(zip(pt_outs, ort_outs)):
        diff = np.max(np.abs(pt_out.numpy() - ort_out))
        assert diff <= 1e-4, f"Parity mismatch on Head {i} for batch {batch_size}: diff={diff:.6e}"
