"""Why is VLMPipeline ~2x faster than our raw infer-request bench?

Hypothesis from reading genai sources (lm_encoding.cpp:183): VLMPipeline runs
prefill as ONE m_llm.infer() call on the full prompt (780 tokens), while
bench_v2/v3 chunked at 128. Four small infers should be slower than one big
one because of:
  - per-call dispatch overhead (×N),
  - FC brgemm utilization at small M (×N short brgemm calls vs one tall one),
  - for the custom op: per-call heap scratch alloc and state read/write (×N),
  - oneDNN cache miss on the smaller shape.

This script holds everything else constant and varies ONLY chunk size, on
both baseline (no rewrites, no .so) and v3 (custom op .so), for prompt_len=512:

    chunk=512   one-shot prefill           (what VLMPipeline effectively does)
    chunk=256
    chunk=128                              (what bench_v3.py does today)
    chunk=64
    chunk=32

If the gap collapses at chunk=512, chunking is the whole story.
If it persists, there's a separate VLMPipeline effect to chase.

Run:
    cd study/qwen3
    QWEN3_USE_C=1 python3 scripts/working/chunk_sweep.py
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))

import numpy as np
import openvino as ov
from fused_linear_attn import (
    register as rla,
    replace_gated_delta_rule_loops_v3,
)
from lm_head_slice import slice_lm_head_to_last_token

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
SO = str(Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so")


def build_baseline(out):
    m = ov.Core().read_model(str(ORIG / "openvino_language_model.xml"))
    slice_lm_head_to_last_token(m)
    ov.serialize(m, out, out.replace(".xml", ".bin"))


def build_v3(out):
    c = ov.Core(); rla(c)
    m = c.read_model(str(ORIG / "openvino_language_model.xml"))
    replace_gated_delta_rule_loops_v3(m)
    slice_lm_head_to_last_token(m)
    ov.serialize(m, out, out.replace(".xml", ".bin"))


def time_prefill(label, xml, with_ext, prompt_len, chunk, repeats=3):
    core = ov.Core()
    if with_ext:
        core.add_extension(SO)
    lm = core.read_model(xml)
    embed = core.compile_model(f"{ORIG}/openvino_text_embeddings_model.xml", "CPU")
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})

    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    rng = np.random.default_rng(0)
    ids = rng.integers(1, 200000, size=(1, prompt_len), dtype=np.int64)

    def embd(x):
        return list(embed.create_infer_request().infer({0: x}).values())[0]

    times = []
    for _ in range(repeats):
        req = compiled.create_infer_request()
        past = 0
        t0 = time.time()
        for i in range(0, prompt_len, chunk):
            L = min(chunk, prompt_len - i)
            req.infer({
                "inputs_embeds": embd(ids[:, i:i+L]),
                "attention_mask": np.ones((1, past + L), dtype=np.int64),
                "position_ids": np.tile(np.arange(past, past + L, dtype=np.int64).reshape(1, 1, L),
                                        (pid_b, 1, 1)),
                "beam_idx": np.zeros((1,), dtype=np.int32),
            })
            past += L
        times.append(time.time() - t0)
        del req
    return min(times)  # best of N


def main():
    print("[build] baseline + v3 IRs…")
    xml_b = "/tmp/lm_baseline.xml";  build_baseline(xml_b)
    xml_v = "/tmp/lm_v3.xml";        build_v3(xml_v)

    PROMPT_LEN = 512
    CHUNKS = [512, 256, 128, 64, 32]

    print(f"\nprompt_len={PROMPT_LEN}, best-of-3, threads=4\n")
    print(f"{'chunk':>6s}  {'baseline (s)':>14s}  {'baseline (tok/s)':>18s}   "
          f"{'v3 (s)':>10s}  {'v3 (tok/s)':>12s}   "
          f"{'baseline/v3':>12s}")

    rows = []
    for ck in CHUNKS:
        # warmup
        time_prefill("warm", xml_b, False, PROMPT_LEN, ck, repeats=1)
        time_prefill("warm", xml_v, True,  PROMPT_LEN, ck, repeats=1)
        tb = time_prefill("base", xml_b, False, PROMPT_LEN, ck, repeats=3)
        tv = time_prefill("v3",   xml_v, True,  PROMPT_LEN, ck, repeats=3)
        rows.append((ck, tb, tv))
        print(f"{ck:>6d}  {tb:>14.3f}  {PROMPT_LEN/tb:>18.1f}   "
              f"{tv:>10.3f}  {PROMPT_LEN/tv:>12.1f}   "
              f"{tb/tv:>12.2f}")


if __name__ == "__main__":
    main()
