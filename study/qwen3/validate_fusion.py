"""
Validate that replacing the gated-delta-rule Loop with the fused GatedDeltaRule
op leaves the language model numerically equivalent.

Compares logits from:
  - original IR (Loop fallback)
  - rewritten IR (custom op via evaluate())

on the same synthetic prefill input.

Usage:
    python validate_fusion.py [--model <ir-dir>] [--prompt-len 16]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import openvino as ov

sys.path.insert(0, str(Path(__file__).parent))
from fused_linear_attn import register, replace_gated_delta_rule_loops  # noqa: E402

HIDDEN = 1024


def make_inputs(seq: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "inputs_embeds": (rng.standard_normal((1, seq, HIDDEN)) * 0.02).astype(np.float32),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
        "position_ids": np.tile(np.arange(seq, dtype=np.int64).reshape(1, 1, seq), (4, 1, 1)),
        "beam_idx": np.zeros((1,), dtype=np.int32),
    }


def run_once(model_xml: str, apply_fusion: bool, inputs: dict):
    core = ov.Core()
    if apply_fusion:
        register(core)
    model = core.read_model(model_xml)
    if apply_fusion:
        n = replace_gated_delta_rule_loops(model)
        print(f"  → replaced {n} Loop(s) with GatedDeltaRule")
    compiled = core.compile_model(model, "CPU", {
        "PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 4,
    })
    req = compiled.create_infer_request()
    req.infer(inputs)
    logits_out = next(o for o in compiled.outputs if "logits" in o.get_any_name())
    return np.asarray(req.get_tensor(logits_out).data).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/qwen3-work/qwen35-0.8b-int8/openvino_language_model.xml")
    ap.add_argument("--prompt-len", type=int, default=16)
    args = ap.parse_args()

    inputs = make_inputs(args.prompt_len)

    print(f"=== Baseline (Loop fallback) — prompt_len={args.prompt_len} ===")
    base = run_once(args.model, apply_fusion=False, inputs=inputs)
    print(f"  logits shape: {base.shape}, dtype: {base.dtype}")

    print(f"\n=== Fused (GatedDeltaRule) — prompt_len={args.prompt_len} ===")
    fused = run_once(args.model, apply_fusion=True, inputs=inputs)
    print(f"  logits shape: {fused.shape}, dtype: {fused.dtype}")

    abs_diff = np.abs(base - fused)
    rel_diff = abs_diff / (np.abs(base) + 1e-6)

    print("\n=== Comparison ===")
    print(f"  max abs diff:  {abs_diff.max():.3e}")
    print(f"  mean abs diff: {abs_diff.mean():.3e}")
    print(f"  max rel diff:  {rel_diff.max():.3e}")
    print(f"  mean rel diff: {rel_diff.mean():.3e}")
    print(f"  base range:    [{base.min():.2f}, {base.max():.2f}]")

    # Argmax overlap on the last position (what sampling actually uses).
    last_base = base[0, -1].argsort()[::-1][:10]
    last_fused = fused[0, -1].argsort()[::-1][:10]
    print(f"\n  top-10 token ids @ last pos:")
    print(f"    baseline: {last_base.tolist()}")
    print(f"    fused:    {last_fused.tolist()}")
    print(f"    overlap:  {len(set(last_base.tolist()) & set(last_fused.tolist()))}/10")

    ok = abs_diff.max() < 1e-2
    print(f"\n{'PASS' if ok else 'FAIL'} (threshold 1e-2)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
