"""Run one (config, attention_backend) pair per process; results go to stdout.

Driven by run_probe_pa_path.sh. Re-uses the fused-dir convention from
bench_genai.py."""
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
    register as register_classes,
    replace_gated_delta_rule_loops,
    replace_gated_delta_rule_loops_v2,
    replace_gated_delta_rule_loops_v3,
)
from lm_head_slice import slice_lm_head_to_last_token

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
SO = Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so"
WORK = Path("/tmp/qwen3-work")
CACHE_DIR = WORK / "genai-cache-probe"
IMG = "/tmp/llama.cpp/media/llama1-logo.png"

LM_FILES = {"openvino_language_model.xml", "openvino_language_model.bin"}

REWRITE = {
    "v1": replace_gated_delta_rule_loops,
    "v2": replace_gated_delta_rule_loops_v2,
    "v3": replace_gated_delta_rule_loops_v3,
}


def load_image(path):
    img = Image.open(path).convert("RGB")
    return ov.Tensor(np.asarray(img, dtype=np.uint8)[None])


def make_fused(version):
    fused = WORK / f"qwen35-0.8b-int8-fused-{version}probe"
    if fused.exists():
        shutil.rmtree(fused)
    fused.mkdir(parents=True, exist_ok=True)
    for f in ORIG.iterdir():
        if f.name in LM_FILES:
            continue
        dst = fused / f.name
        dst.symlink_to(f, target_is_directory=f.is_dir())
    c = ov.Core(); register_classes(c)
    m = c.read_model(str(ORIG / "openvino_language_model.xml"))
    n = REWRITE[version](m)
    slice_lm_head_to_last_token(m)
    print(f"  {version} rewrite: replaced={n}")
    ov.serialize(m, str(fused / "openvino_language_model.xml"),
                 str(fused / "openvino_language_model.bin"))
    return fused


def run(model_dir, props, label):
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    props = dict(props)
    props["CACHE_DIR"] = str(CACHE_DIR)
    props["INFERENCE_NUM_THREADS"] = 4
    image = load_image(IMG)
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 32
    cfg.do_sample = False

    t0 = time.time()
    vlm = ov_genai.VLMPipeline(str(model_dir), "CPU", **props)
    load = time.time() - t0
    vlm.generate("Describe this image.", images=[image], generation_config=cfg)  # warmup
    out = vlm.generate("Describe this image.", images=[image], generation_config=cfg)
    pm = out.perf_metrics
    ttft = pm.get_ttft().mean / 1000.0
    tput = pm.get_throughput().mean
    n_in = pm.get_num_input_tokens()
    text = str(out).strip().replace("\n", " ")[:80]
    print(f"\n=== {label} ===")
    print(f"  load:        {load:6.2f}s")
    print(f"  in tokens:   {n_in}")
    print(f"  TTFT:        {ttft:6.3f}s   ({n_in/ttft:6.1f} tok/s)")
    print(f"  decode:      {tput:6.2f} tok/s")
    print(f"  output:      {text!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["baseline", "v1", "v2", "v3"])
    ap.add_argument("--backend", required=True, choices=["pa", "sdpa"])
    args = ap.parse_args()
    print(f"openvino={ov.__version__}  openvino_genai={ov_genai.__version__}")
    print(f"version={args.version}  backend={args.backend}")

    if args.version == "baseline":
        model_dir = ORIG
        ext_props = {}
    else:
        model_dir = make_fused(args.version)
        ext_props = {"extensions": [str(SO)]}

    if args.backend == "sdpa":
        ext_props["ATTENTION_BACKEND"] = "SDPA"

    run(model_dir, ext_props, f"{args.version}_{args.backend}")


if __name__ == "__main__":
    main()
