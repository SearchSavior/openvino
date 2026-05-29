"""
Export a fused-IR variant with a configurable subset of rewrites applied.

Examples:
  python export_fused.py --rewrites gdr conv1d lm_head_slice \\
      --out /tmp/qwen3-work/qwen35-0.8b-int8-fused
  python export_fused.py --rewrites gdr lm_head_slice \\
      --out /tmp/qwen3-work/qwen35-0.8b-int8-fused-light
  python export_fused.py --rewrites lm_head_slice \\
      --out /tmp/qwen3-work/qwen35-0.8b-int8-slice-only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import openvino as ov
from fused_linear_attn import replace_gated_delta_rule_loops
from fused_conv1d import replace_causal_conv1d_chains
from lm_head_slice import slice_lm_head_to_last_token

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
LM_FILES = {"openvino_language_model.xml", "openvino_language_model.bin"}


def link_supporting_files(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for f in ORIG.iterdir():
        dst = out / f.name
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                continue
        if f.name in LM_FILES:
            continue
        dst.symlink_to(f, target_is_directory=f.is_dir())


def apply_rewrites(model, rewrites):
    counts = {"gdr": 0, "conv1d": 0, "lm_head_slice": False}
    if "gdr" in rewrites:
        counts["gdr"] = replace_gated_delta_rule_loops(model)
    if "conv1d" in rewrites:
        counts["conv1d"] = replace_causal_conv1d_chains(model)
    if "lm_head_slice" in rewrites:
        counts["lm_head_slice"] = slice_lm_head_to_last_token(model)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites", nargs="+", required=True,
                    choices=["gdr", "conv1d", "lm_head_slice"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    link_supporting_files(out)

    model = ov.Core().read_model(str(ORIG / "openvino_language_model.xml"))
    counts = apply_rewrites(model, args.rewrites)
    ov.serialize(model, str(out / "openvino_language_model.xml"),
                 str(out / "openvino_language_model.bin"))
    print(f"rewrites applied: {counts}")
    print(f"serialized → {out}/openvino_language_model.xml")


if __name__ == "__main__":
    main()
