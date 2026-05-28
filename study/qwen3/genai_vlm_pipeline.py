"""
Drive Qwen3.5-VL through ov_genai.VLMPipeline with our fused custom ops loaded
as a proper OpenVINO C++ extension library (cpp_ext/build/libqwen3_ov_ext.so).

Pipeline:
  1. Read the original LM IR
  2. Apply the three graph rewrites (gdr + conv1d + lm_head slice)
  3. Serialize a fused model dir (other files symlinked)
  4. VLMPipeline(fused_dir, "CPU", extensions=[str(.so)]) — registers the
     custom op factories at the extension layer
  5. Generate text via the high-level genai API

Also runs the same thing through plain ov.Core() to confirm the .so on its own.

Requires openvino_genai >= 2026.3.x dev nightly — the extensions= property in
the LLM/VLMPipeline binding shipped in the 2026.3 series. Install with:
    uv pip install --system --pre -U openvino-genai \\
        --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import openvino as ov
import openvino_genai as ov_genai

from fused_linear_attn import replace_gated_delta_rule_loops
from fused_conv1d import replace_causal_conv1d_chains
from lm_head_slice import slice_lm_head_to_last_token

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
FUSED = Path("/tmp/qwen3-work/qwen35-0.8b-int8-fused")
SO_PATH = Path(__file__).parent / "cpp_ext/build/libqwen3_ov_ext.so"

LM_FILES = {"openvino_language_model.xml", "openvino_language_model.bin"}


def make_fused_dir():
    FUSED.mkdir(parents=True, exist_ok=True)
    for f in ORIG.iterdir():
        dst = FUSED / f.name
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                continue
        if f.name in LM_FILES:
            continue
        dst.symlink_to(f, target_is_directory=f.is_dir())


def rewrite_and_save():
    model = ov.Core().read_model(str(ORIG / "openvino_language_model.xml"))
    n1 = replace_gated_delta_rule_loops(model)
    n2 = replace_causal_conv1d_chains(model)
    ok = slice_lm_head_to_last_token(model)
    print(f"  rewrites: linear_attn={n1}  conv1d={n2}  lm_head_slice={ok}")
    ov.serialize(model, str(FUSED / "openvino_language_model.xml"),
                 str(FUSED / "openvino_language_model.bin"))


def try_genai_extensions():
    prompt = "What is your show size?"

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 32

    print("\n=== (A) VLMPipeline + extensions=[str(.so)] + generate ===")
    t0 = time.time()
    vlm = ov_genai.VLMPipeline(str(FUSED), "CPU", extensions=[str(SO_PATH)])
    print(f"  loaded in {time.time() - t0:.2f}s")
    t0 = time.time()
    out = vlm.generate(prompt, generation_config=cfg)
    print(f"  generated in {time.time() - t0:.2f}s")
    print(f"  output: {str(out)!r}")
    del vlm

    print("\n=== (B) sanity: extensions=[bogus_path] should error ===")
    try:
        ov_genai.LLMPipeline(str(FUSED), "CPU", extensions=["/tmp/does_not_exist.so"])
        print("  unexpectedly loaded — extension path not validated")
    except RuntimeError as e:
        msg = str(e).replace("\n", " ")[:200]
        print(f"  errored as expected: {msg}")


def run_via_ov_core():
    from transformers import AutoTokenizer
    print("\n=== end-to-end via ov.Core().add_extension(...) (working path) ===")
    core = ov.Core()
    core.add_extension(str(SO_PATH))
    embed = core.compile_model(str(FUSED / "openvino_text_embeddings_model.xml"), "CPU")
    lm = core.read_model(str(FUSED / "openvino_language_model.xml"))
    n_gdr = sum(1 for n in lm.get_ops() if n.get_type_name() == "GatedDeltaRule")
    n_cv = sum(1 for n in lm.get_ops() if n.get_type_name() == "FusedCausalConv1d")
    print(f"  custom ops resolved: GatedDeltaRule={n_gdr}  FusedCausalConv1d={n_cv}")

    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})
    req = compiled.create_infer_request()
    logits_out = next(o for o in compiled.outputs if "logits" in o.get_any_name())

    tok = AutoTokenizer.from_pretrained(str(FUSED))
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "What is your show size?"}],
        tokenize=False, add_generation_prompt=True)
    ids = np.asarray([tok.encode(prompt)], dtype=np.int64)
    T = ids.shape[1]

    t0 = time.time()
    req.infer({
        "inputs_embeds": list(embed.create_infer_request().infer({0: ids}).values())[0],
        "attention_mask": np.ones((1, T), dtype=np.int64),
        "position_ids": np.tile(np.arange(T, dtype=np.int64).reshape(1, 1, T), (4, 1, 1)),
        "beam_idx": np.zeros((1,), dtype=np.int32),
    })
    print(f"  prefill: {time.time() - t0:.2f}s ({T} tokens)")

    next_id = int(np.asarray(req.get_tensor(logits_out).data)[0, -1].argmax())
    gen = [next_id]
    past = T
    t1 = time.time()
    for _ in range(31):
        ne = list(embed.create_infer_request().infer({0: np.array([[next_id]], dtype=np.int64)}).values())[0]
        req.infer({
            "inputs_embeds": ne,
            "attention_mask": np.ones((1, past + 1), dtype=np.int64),
            "position_ids": np.full((4, 1, 1), past, dtype=np.int64),
            "beam_idx": np.zeros((1,), dtype=np.int32),
        })
        next_id = int(np.asarray(req.get_tensor(logits_out).data)[0, -1].argmax())
        gen.append(next_id)
        past += 1
    print(f"  decode 32 toks: {time.time() - t1:.2f}s")
    print(f"  output: {tok.decode(gen, skip_special_tokens=True)!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-genai", action="store_true", help="skip the (failing) genai probe")
    args = ap.parse_args()

    print(f"openvino={ov.__version__}")
    print(f"openvino_genai={ov_genai.__version__}")
    if not SO_PATH.exists():
        sys.exit(f"missing {SO_PATH} — build with cpp_ext/build_kernels invocation in CMakeLists")

    print("\n[1] preparing fused IR")
    make_fused_dir()
    rewrite_and_save()

    if not args.skip_genai:
        try_genai_extensions()
    run_via_ov_core()


if __name__ == "__main__":
    main()
