"""Confirm or reject the hypothesis: VLMPipeline's baseline runs on
PagedAttention (PA_BACKEND -> VLMContinuousBatchingAdapter), but our v3
custom-op model falls back to stateful (VLMPipelineImpl) because PA
transformations can't rewrite the GatedDeltaRule op.

Test matrix (single invocation per row, fresh cache each, cold load):

    baseline   default (PA_BACKEND)                expect fast
    baseline   ATTENTION_BACKEND=SDPA              expect slow  <- proves PA is the speedup
    v3         default (PA_BACKEND, falls back?)   expect slow
    v3         ATTENTION_BACKEND=SDPA              expect same  <- proves no PA was active

If row 2 prints "fast" then PA is not the differentiator and we have to
keep looking; if row 2 prints "slow" then we've isolated it.

Run:
    cd study/qwen3
    QWEN3_USE_C=1 python3 scripts/working/probe_pa_path.py --config baseline_pa
    QWEN3_USE_C=1 python3 scripts/working/probe_pa_path.py --config baseline_sdpa
    QWEN3_USE_C=1 python3 scripts/working/probe_pa_path.py --config v3_pa
    QWEN3_USE_C=1 python3 scripts/working/probe_pa_path.py --config v3_sdpa
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

from fused_linear_attn import replace_gated_delta_rule_loops_v3
from lm_head_slice import slice_lm_head_to_last_token

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
SO = Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so"
WORK = Path("/tmp/qwen3-work")
CACHE_DIR = WORK / "genai-cache-probe"
IMG = "/tmp/llama.cpp/media/llama1-logo.png"


def load_image(path):
    img = Image.open(path).convert("RGB")
    return ov.Tensor(np.asarray(img, dtype=np.uint8)[None])


def make_fused_v3():
    fused = WORK / "qwen35-0.8b-int8-fused-v3probe"
    fused.mkdir(parents=True, exist_ok=True)
    LM = {"openvino_language_model.xml", "openvino_language_model.bin"}
    for f in ORIG.iterdir():
        if f.name in LM:
            continue
        dst = fused / f.name
        if dst.is_symlink() or dst.exists(): dst.unlink()
        dst.symlink_to(f, target_is_directory=f.is_dir())
    m = ov.Core().read_model(str(ORIG / "openvino_language_model.xml"))
    n = replace_gated_delta_rule_loops_v3(m)
    slice_lm_head_to_last_token(m)
    print(f"  v3 rewrite: replaced={n}")
    ov.serialize(m, str(fused / "openvino_language_model.xml"),
                 str(fused / "openvino_language_model.bin"))
    return fused


def run(model_dir, props, label):
    if CACHE_DIR.exists(): shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    props = dict(props); props["CACHE_DIR"] = str(CACHE_DIR)
    props["INFERENCE_NUM_THREADS"] = 4
    image = load_image(IMG)
    cfg = ov_genai.GenerationConfig(); cfg.max_new_tokens = 32; cfg.do_sample = False

    t0 = time.time()
    vlm = ov_genai.VLMPipeline(str(model_dir), "CPU", **props)
    load = time.time() - t0
    # warmup
    vlm.generate("Describe this image.", images=[image], generation_config=cfg)
    out = vlm.generate("Describe this image.", images=[image], generation_config=cfg)
    pm = out.perf_metrics
    ttft = pm.get_ttft().mean / 1000.0
    tput = pm.get_throughput().mean
    n_in = pm.get_num_input_tokens()
    print(f"\n=== {label} ===")
    print(f"  load:        {load:6.2f}s")
    print(f"  in tokens:   {n_in}")
    print(f"  TTFT:        {ttft:6.3f}s   ({n_in/ttft:6.1f} tok/s)")
    print(f"  decode:      {tput:6.2f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["baseline_pa", "baseline_sdpa", "v3_pa", "v3_sdpa"])
    args = ap.parse_args()
    print(f"openvino={ov.__version__}  openvino_genai={ov_genai.__version__}")
    print(f"config={args.config}")

    if args.config.startswith("baseline"):
        model_dir = ORIG; ext_props = {}
    else:
        model_dir = make_fused_v3(); ext_props = {"extensions": [str(SO)]}

    if args.config.endswith("_sdpa"):
        ext_props["ATTENTION_BACKEND"] = "SDPA"
    # else: default = PA

    run(model_dir, ext_props, args.config)


if __name__ == "__main__":
    main()
