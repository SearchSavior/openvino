"""Single-config genai vision-language bench for one fusion variant.

Runs exactly ONE configuration per process invocation. Drive it from
run_bench_genai.sh to sequence baseline/v1/v2/v3 in fresh processes; the
compile cache (CACHE_DIR) is wiped before each load.

Configs:
  baseline  stock VLMPipeline on the original model dir, no rewrites, no .so.
  v1/v2/v3  rewrite the LM IR with the matching GatedDeltaRule fusion,
            serialize a fused model dir, load with extensions=[so].

genai's internal Core only ever sees the .so (no Python Op subclass in its
namespace), so the C++ GatedDeltaRule{,V2,V3} implementation handles
evaluate(). The Python rewrites only construct+serialize the fused IR.

Findings and interpretation live in DISCUSSION.md, not here.

Run one:
    cd study/qwen3
    QWEN3_USE_C=1 python3 scripts/working/bench_genai.py --config v3
Run all:
    QWEN3_USE_C=1 bash scripts/working/run_bench_genai.sh
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image

from fused_linear_attn import (
    replace_gated_delta_rule_loops,
    replace_gated_delta_rule_loops_v2,
    replace_gated_delta_rule_loops_v3,
)
from lm_head_slice import slice_lm_head_to_last_token

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
SO_PATH = Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so"
WORK = Path("/tmp/qwen3-work")
CACHE_DIR = WORK / "genai-cache"

LM_FILES = {"openvino_language_model.xml", "openvino_language_model.bin"}

REWRITE = {
    "v1": replace_gated_delta_rule_loops,
    "v2": replace_gated_delta_rule_loops_v2,
    "v3": replace_gated_delta_rule_loops_v3,
}

PROMPT = "Describe this image in one short sentence, then say who you are."
DEFAULT_IMAGE = "/tmp/llama.cpp/media/llama1-logo.png"


def load_image(path):
    """Load an image as an ov.Tensor of shape [1, H, W, 3] uint8, the layout
    ov_genai.VLMPipeline expects."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)[None]  # [1, H, W, 3]
    return ov.Tensor(arr)


def make_fused_dir(config):
    """Build a per-config fused model dir; symlink everything but the LM,
    then serialize the rewritten LM into it. Returns the dir path."""
    fused = WORK / f"qwen35-0.8b-int8-fused-{config}"
    fused.mkdir(parents=True, exist_ok=True)
    for f in ORIG.iterdir():
        if f.name in LM_FILES:
            continue
        dst = fused / f.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(f, target_is_directory=f.is_dir())

    model = ov.Core().read_model(str(ORIG / "openvino_language_model.xml"))
    n = REWRITE[config](model)
    ok = slice_lm_head_to_last_token(model)
    print(f"  rewrites: replaced={n}  lm_head_slice={ok}")
    ov.serialize(model, str(fused / "openvino_language_model.xml"),
                 str(fused / "openvino_language_model.bin"))
    return fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["baseline", "v1", "v2", "v3"])
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    print(f"openvino={ov.__version__}  openvino_genai={ov_genai.__version__}")
    print(f"config={args.config}  image={args.image}")
    if not Path(args.image).exists():
        sys.exit(f"missing image {args.image} — pass --image <path>")

    # Rebuild the compile cache each run for a consistent cold load time.
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve model dir + extension args per config (duplicated on purpose).
    if args.config == "baseline":
        model_dir = ORIG
        pipe_kwargs = {"CACHE_DIR": str(CACHE_DIR),
                       "INFERENCE_NUM_THREADS": args.threads}
        print("  (stock VLMPipeline, no rewrites, no extensions)")
    else:
        if not SO_PATH.exists():
            sys.exit(f"missing {SO_PATH} — build cpp_ext first")
        model_dir = make_fused_dir(args.config)
        pipe_kwargs = {"CACHE_DIR": str(CACHE_DIR),
                       "INFERENCE_NUM_THREADS": args.threads,
                       "extensions": [str(SO_PATH)]}

    image = load_image(args.image)
    print(f"  image tensor={image.get_shape()}")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = args.max_new_tokens
    cfg.do_sample = False  # greedy, deterministic

    t0 = time.time()
    vlm = ov_genai.VLMPipeline(str(model_dir), "CPU", **pipe_kwargs)
    load_s = time.time() - t0

    # warmup (primes weight-prepack + vision encode; cache already built above)
    vlm.generate(PROMPT, images=[image], generation_config=cfg)

    out = vlm.generate(PROMPT, images=[image], generation_config=cfg)
    pm = out.perf_metrics

    ttft = pm.get_ttft().mean / 1000.0           # ms -> s
    tput = pm.get_throughput().mean              # generated tok/s
    tpot = pm.get_tpot().mean                     # ms/token
    n_in = pm.get_num_input_tokens()
    n_gen = pm.get_num_generated_tokens()
    gen_total = pm.get_generate_duration().mean / 1000.0
    text = str(out).strip().replace("\n", " ")

    print(f"\n=== {args.config} (genai VLMPipeline + image) ===")
    print(f"  load (cold cache): {load_s:6.2f}s")
    print(f"  input tokens:      {n_in}")
    print(f"  generated:         {n_gen}")
    print(f"  TTFT (vis+pp):     {ttft:6.3f}s  ({n_in/ttft:6.1f} tok/s)")
    print(f"  decode TPOT:       {tpot:6.2f} ms/tok ({1000.0/tpot:5.2f} tok/s)")
    print(f"  decode tput:       {tput:6.2f} tok/s")
    print(f"  generate total:    {gen_total:6.2f}s")
    print(f"  output: {text[:100]!r}")


if __name__ == "__main__":
    main()
