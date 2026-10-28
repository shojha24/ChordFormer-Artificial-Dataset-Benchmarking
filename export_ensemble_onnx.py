#!/usr/bin/env python3
"""
export_ensemble_onnx.py

Fuses the 5 trained ChordFormer fold checkpoints into a single unified
PyTorch nn.Module (EnsembleChordFormer), exports the graph to an optimized
ONNX model with dynamic batching, and validates numerical parity against
ONNX Runtime across multiple batch sizes.

Target input:
  cqt_input: [B, 431, 252] (10.0s audio segment @ 22,050 Hz, hop length 512, 252 CQT bins)

Target outputs (6 structural heads, ensemble softmax probabilities):
  root_triad: [B, 431, 97]
  bass:       [B, 431, 13]
  seventh:    [B, 431, 4]
  ninth:      [B, 431, 4]
  eleventh:   [B, 431, 3]
  thirteenth: [B, 431, 3]
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure current repository is in sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from confor_head16 import ChordNet


class EnsembleChordFormer(nn.Module):
    """
    Unified 5-fold Ensemble wrapper module.
    Fuses 5 ChordFormer fold models into a single graph to eliminate Triton
    multi-model scheduling overhead and allow TensorRT stream/kernel fusion.
    """

    def __init__(self, checkpoints: List[str], device: str = "cpu"):
        super().__init__()
        self.models = nn.ModuleList()
        print(f"[EnsembleChordFormer] Loading {len(checkpoints)} fold checkpoints onto {device}...")
        for i, ckpt_path in enumerate(checkpoints):
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            sd = torch.load(ckpt_path, map_location=device)
            net_state = sd["net"] if "net" in sd else sd
            model = ChordNet(None)
            model.load_state_dict(net_state, strict=False)
            model.eval()
            self.models.append(model)
            print(f"  - Loaded fold {i} from: {os.path.basename(ckpt_path)}")

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): CQT features with shape [B, 431, 252].

        Returns:
            Tuple of 6 tensors containing mean ensemble probabilities:
              root_triad: [B, 431, 97]
              bass:       [B, 431, 13]
              seventh:    [B, 431, 4]
              ninth:      [B, 431, 4]
              eleventh:   [B, 431, 3]
              thirteenth: [B, 431, 3]
        """
        # Feature padding: pad last dimension from 252 to 256 for 16-head Conformer alignment
        padding = (0, 4, 0, 0, 0, 0)
        x_pad = F.pad(x, padding, "constant", 0.0)

        p_root = 0.0
        p_bass = 0.0
        p_7 = 0.0
        p_9 = 0.0
        p_11 = 0.0
        p_13 = 0.0

        for model in self.models:
            # chordfor forward: [B, 431, 256] -> [B, 431, 124]
            out = model.chordfor(x_pad)
            p_root = p_root + F.softmax(out[:, :, :97], dim=-1)
            p_bass = p_bass + F.softmax(out[:, :, 97:110], dim=-1)
            p_7 = p_7 + F.softmax(out[:, :, 110:114], dim=-1)
            p_9 = p_9 + F.softmax(out[:, :, 114:118], dim=-1)
            p_11 = p_11 + F.softmax(out[:, :, 118:121], dim=-1)
            p_13 = p_13 + F.softmax(out[:, :, 121:124], dim=-1)

        scale = 1.0 / len(self.models)
        return (
            p_root * scale,
            p_bass * scale,
            p_7 * scale,
            p_9 * scale,
            p_11 * scale,
            p_13 * scale,
        )


def export_onnx(
    ensemble: EnsembleChordFormer,
    output_path: str,
    opset_version: int = 17,
    device: str = "cpu",
) -> None:
    """
    Exports the EnsembleChordFormer PyTorch module to an ONNX graph with
    a dynamic batch dimension on dim 0 and fixed sequence/feature dimensions.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ensemble.to(device)
    ensemble.eval()

    dummy_input = torch.randn(2, 431, 252, device=device)

    input_names = ["cqt_input"]
    output_names = [
        "root_triad",
        "bass",
        "seventh",
        "ninth",
        "eleventh",
        "thirteenth",
    ]

    dynamic_axes = {
        "cqt_input": {0: "batch_size"},
        "root_triad": {0: "batch_size"},
        "bass": {0: "batch_size"},
        "seventh": {0: "batch_size"},
        "ninth": {0: "batch_size"},
        "eleventh": {0: "batch_size"},
        "thirteenth": {0: "batch_size"},
    }

    print(f"\n[Export] Exporting ONNX model to: {output_path} (opset {opset_version})...")
    t0 = time.time()
    torch.onnx.export(
        ensemble,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Export] Succeeded in {elapsed:.2f}s! File size: {file_size_mb:.2f} MB")


def verify_parity(
    ensemble: EnsembleChordFormer,
    onnx_path: str,
    batch_sizes: List[int] = [1, 4, 8],
    device: str = "cpu",
    atol: float = 1e-4,
) -> bool:
    """
    Verifies numerical output parity between PyTorch and ONNX Runtime
    across multiple batch sizes.
    """
    import onnxruntime as ort

    print(f"\n[Verification] Loading ONNX Runtime session for: {onnx_path}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

    all_passed = True
    head_names = ["root_triad", "bass", "seventh", "ninth", "eleventh", "thirteenth"]

    ensemble.to(device)
    ensemble.eval()

    for b in batch_sizes:
        print(f"\n--- Testing Batch Size B = {b} (Shape: [{b}, 431, 252]) ---")
        x_torch = torch.randn(b, 431, 252, device=device)
        with torch.no_grad():
            pt_outs = ensemble(x_torch)

        x_np = x_torch.cpu().numpy()
        ort_outs = sess.run(None, {"cqt_input": x_np})

        batch_max_err = 0.0
        for i, (head_name, pt_out, ort_out) in enumerate(zip(head_names, pt_outs, ort_outs)):
            pt_np = pt_out.cpu().numpy()
            diff = np.max(np.abs(pt_np - ort_out))
            batch_max_err = max(batch_max_err, diff)
            status = "PASS" if diff <= atol else "FAIL"
            print(f"  Head {i} ({head_name:10s}): max absolute error = {diff:.6e} [{status}]")

        print(f"  => B={b} Overall Max Error: {batch_max_err:.6e} (Threshold: {atol:.1e})")
        if batch_max_err > atol:
            all_passed = False

    if all_passed:
        print("\n[Verification Result] ALL PARITY CHECKS PASSED SUCCESSFULLY! (Error <= 1e-4)")
    else:
        print("\n[Verification Result] ERROR: Numerical parity check failed for one or more heads!")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Export 5-fold ChordFormer ensemble to ONNX.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            os.path.join(REPO_ROOT, f"cache_data/chordformer_head16_synth_ft_loss_s{i}.best.sdict")
            for i in range(5)
        ],
        help="Paths to the 5 fold checkpoint state_dicts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(REPO_ROOT, "model_repository/chordformer_trt/1/model.onnx"),
        help="Destination path for the exported ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export and CPU validation ('cpu' recommended).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run ONNX Runtime parity verification against PyTorch.",
    )
    parser.add_argument(
        "--skip-verify",
        dest="verify",
        action="store_false",
        help="Skip ONNX Runtime parity verification.",
    )

    args = parser.parse_args()

    print("======================================================================")
    print("ChordFormer 5-Fold Ensemble ONNX Export & Optimization Pipeline")
    print("======================================================================")
    print(f"Checkpoints ({len(args.checkpoints)} folds):")
    for cp in args.checkpoints:
        print(f"  - {cp}")
    print(f"Target Output: {args.output}")
    print(f"Opset: {args.opset}")
    print(f"Device: {args.device}")

    ensemble = EnsembleChordFormer(args.checkpoints, device=args.device)
    export_onnx(ensemble, args.output, opset_version=args.opset, device=args.device)

    if args.verify:
        success = verify_parity(ensemble, args.output, batch_sizes=[1, 4, 8], device=args.device)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
