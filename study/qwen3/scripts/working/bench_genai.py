"""End-to-end genai *vision-language* bench of v1 / v2 / v3 custom-op fusions.

Unlike bench_v2 / bench_v3 (which drive the raw ov.Core() infer-request loop
on the language model alone and need the serialize/reload trick so the C++ .so
wins evaluate over the Python Op subclass), this script goes through the
high-level `ov_genai.VLMPipeline(fused_dir, "CPU", extensions=[so])` API and
feeds a real **image** so the full multimodal path runs:

    image -> vision_embeddings -> merger -> [image tokens] + text tokens
          -> language_model (with our fused GatedDeltaRule{,V2,V3} ops) -> text

genai's internal Core only ever sees the .so — there is no Python `Op`
subclass registered in its namespace — so the C++ implementation wins evaluate
natively. The Python rewrites are used here only to *construct + serialize* the
fused IR into a model dir; once it's on disk, genai reads it back and the
extension factory resolves the custom op.

For each version we:
  1. read the original LM IR, apply the matching rewrite, serialize into a
     per-version fused dir (other model files symlinked from the original),
  2. load through VLMPipeline(..., extensions=[so]),
  3. generate from an image + question and read genai's PerfMetrics:
       - TTFT  (time to first token ~= vision encode + prefill)
       - throughput (generation tok/s)
       - per-version load time.

This is the apples-to-apples "would a genai VLM user actually see this?"
number, as opposed to bench_v3's controlled text-only pp512/tg32 micro-bench.

Run:
    cd study/qwen3
    QWEN3_USE_C=1 python3 scripts/working/bench_genai.py [--image /path.png]
"""
import argparse
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

LM_FILES = {"openvino_language_model.xml", "openvino_language_model.bin"}

REWRITE = {
    1: replace_gated_delta_rule_loops,
    2: replace_gated_delta_rule_loops_v2,
    3: replace_gated_delta_rule_loops_v3,
}

# Vision-language prompt. The image tokens dominate the prefill; the question
# is short on purpose so TTFT reflects vision-encode + image-prefill cost.
PROMPT = "Describe this image in one short sentence, then say who you are."
DEFAULT_IMAGE = "/tmp/llama.cpp/media/llama1-logo.png"


def load_image(path):
    """Load an image as an ov.Tensor of shape [1, H, W, 3] uint8, the layout
    ov_genai.VLMPipeline expects."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)[None]  # [1, H, W, 3]
    return ov.Tensor(arr)


def make_fused_dir(version):
    """Build a per-version fused model dir; symlink everything but the LM,
    then serialize the rewritten LM into it. Returns the dir path."""
    fused = WORK / f"qwen35-0.8b-int8-fused-v{version}"
    fused.mkdir(parents=True, exist_ok=True)
    for f in ORIG.iterdir():
        if f.name in LM_FILES:
            continue
        dst = fused / f.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(f, target_is_directory=f.is_dir())

    model = ov.Core().read_model(str(ORIG / "openvino_language_model.xml"))
    n = REWRITE[version](model)
    ok = slice_lm_head_to_last_token(model)
    print(f"  v{version} rewrites: replaced={n}  lm_head_slice={ok}")
    ov.serialize(model, str(fused / "openvino_language_model.xml"),
                 str(fused / "openvino_language_model.bin"))
    return fused


def bench(version, image, max_new_tokens=32):
    fused = make_fused_dir(version)

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False  # greedy, deterministic, comparable to bench_v3

    t0 = time.time()
    vlm = ov_genai.VLMPipeline(str(fused), "CPU", extensions=[str(SO_PATH)],
                               INFERENCE_NUM_THREADS=4)
    load_s = time.time() - t0

    # warmup (also primes any lazy compile / weight-prepack + vision encode)
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
    del vlm

    print(f"\n=== v{version} (genai VLMPipeline + image + extensions=[.so]) ===")
    print(f"  load:            {load_s:6.2f}s")
    print(f"  input tokens:    {n_in}")
    print(f"  generated:       {n_gen}")
    print(f"  TTFT (vis+pp):   {ttft:6.3f}s  ({n_in/ttft:6.1f} tok/s)")
    print(f"  decode TPOT:     {tpot:6.2f} ms/tok ({1000.0/tpot:5.2f} tok/s)")
    print(f"  decode tput:     {tput:6.2f} tok/s")
    print(f"  generate total:  {gen_total:6.2f}s")
    print(f"  output: {text[:90]!r}")
    return {"version": version, "load": load_s, "n_in": n_in, "n_gen": n_gen,
            "ttft": ttft, "pp_tps": n_in / ttft, "tpot": tpot,
            "tg_tps": tput, "gen_total": gen_total, "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    args = ap.parse_args()

    print(f"openvino={ov.__version__}")
    print(f"openvino_genai={ov_genai.__version__}")
    if not SO_PATH.exists():
        sys.exit(f"missing {SO_PATH} — build cpp_ext first "
                 f"(cd cpp_ext && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j)")
    if not Path(args.image).exists():
        sys.exit(f"missing image {args.image} — pass --image <path>")

    image = load_image(args.image)
    print(f"image={args.image}  tensor={image.get_shape()}")
    rows = [bench(v, image) for v in (1, 2, 3)]

    print(f"\n{'='*78}\nSUMMARY (genai VLMPipeline, image+text, greedy, threads=4)\n{'='*78}")
    print(f"{'metric':<22s} {'v1':>16s} {'v2':>16s} {'v3':>16s}")
    for label, key, unit, fmt in [
        ("load",            "load",      "s",     "{:.2f}"),
        ("input tokens",    "n_in",      "",      "{:.0f}"),
        ("TTFT prefill",    "ttft",      "s",     "{:.3f}"),
        ("prefill tput",    "pp_tps",    "tok/s", "{:.1f}"),
        ("decode TPOT",     "tpot",      "ms",    "{:.2f}"),
        ("decode tput",     "tg_tps",    "tok/s", "{:.2f}"),
        ("generate total",  "gen_total", "s",     "{:.2f}"),
    ]:
        row = f"  {label:<20s}"
        for r in rows:
            row += f" {fmt.format(r[key]):>10s} {unit:<5s}"
        print(row)

    same = len({r["text"] for r in rows}) == 1
    print(f"\noutputs identical across versions: {same}")
    if not same:
        for r in rows:
            print(f"  v{r['version']}: {r['text'][:80]!r}")


if __name__ == "__main__":
    main()
