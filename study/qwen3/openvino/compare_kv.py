"""
Run two generation passes (unfused vs fully fused) and compare persistent state.

For each config:
  - Run prefill on a chat-templated prompt, snapshot all stateful tensors.
  - Run 64 decode steps, snapshot again.
  - Sum bytes by state type (linear-attn / conv1d / full-attn KV) and per-token growth.

The three rewrites (linear-attn fusion, conv1d fusion, lm_head slice) target
working memory and compute, not the persistent state — this script verifies
that claim by reading the actual state tensors via query_state().
"""
import sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernels"))
import numpy as np
import openvino as ov
from transformers import AutoTokenizer
from fused_linear_attn import register as register_la, replace_gated_delta_rule_loops
from fused_conv1d import register as register_cv, replace_causal_conv1d_chains
from lm_head_slice import slice_lm_head_to_last_token

MODEL_DIR = "/tmp/qwen3-work/qwen35-0.8b-int8"
PROMPT = "What is your show size?"
MAX_NEW_TOKENS = 64


def classify_state(shape):
    """Bucket a state tensor by shape -> human-readable category."""
    if len(shape) == 4 and shape[1] == 16 and shape[2] == 128 and shape[3] == 128:
        return "linear_attn_state"
    if len(shape) == 3 and shape[1] == 6144:
        return "conv1d_state"
    if len(shape) == 4 and shape[1] == 2 and shape[3] == 256:
        return "full_attn_kv"
    return f"other_{shape}"


def snapshot_state(req):
    """Return list of (category, shape, bytes) for every current state tensor."""
    out = []
    for s in req.query_state():
        t = s.state
        sh = tuple(t.shape)
        nbytes = int(np.prod(sh)) * t.element_type.size
        out.append((classify_state(sh), sh, nbytes))
    return out


def fmt_bytes(b):
    if b >= 1024 * 1024:
        return f"{b / 1024 / 1024:8.2f} MB"
    if b >= 1024:
        return f"{b / 1024:8.2f} KB"
    return f"{b:6d}   B"


def aggregate(snapshot):
    by_cat = defaultdict(lambda: [0, 0])  # [count, total_bytes]
    total = 0
    for cat, sh, b in snapshot:
        by_cat[cat][0] += 1
        by_cat[cat][1] += b
        total += b
    return by_cat, total


def run(label, apply_fusions):
    print(f"\n{'=' * 64}\n{label}\n{'=' * 64}")
    core = ov.Core()
    if apply_fusions:
        register_la(core); register_cv(core)
    embed_model = core.compile_model(f"{MODEL_DIR}/openvino_text_embeddings_model.xml", "CPU")
    lm = core.read_model(f"{MODEL_DIR}/openvino_language_model.xml")
    if apply_fusions:
        print(f"  linear-attn replaced: {replace_gated_delta_rule_loops(lm)}")
        print(f"  conv1d replaced:      {replace_causal_conv1d_chains(lm)}")
        print(f"  lm_head slice:        {slice_lm_head_to_last_token(lm)}")
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})
    req = compiled.create_infer_request()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    input_ids = np.asarray([tok.encode(prompt_text)], dtype=np.int64)
    T = input_ids.shape[1]

    def embed(ids):
        return list(embed_model.create_infer_request().infer({0: ids}).values())[0]

    logits_out = next(o for o in compiled.outputs if "logits" in o.get_any_name())

    # Prefill
    req.infer({
        "inputs_embeds": embed(input_ids),
        "attention_mask": np.ones((1, T), dtype=np.int64),
        "position_ids": np.tile(np.arange(T, dtype=np.int64).reshape(1, 1, T), (4, 1, 1)),
        "beam_idx": np.zeros((1,), dtype=np.int32),
    })
    snap_prefill = snapshot_state(req)
    next_id = int(np.asarray(req.get_tensor(logits_out).data)[0, -1].argmax())

    # Decode
    past_len = T
    for _ in range(MAX_NEW_TOKENS - 1):
        ne = embed(np.array([[next_id]], dtype=np.int64))
        req.infer({
            "inputs_embeds": ne,
            "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
            "position_ids": np.full((4, 1, 1), past_len, dtype=np.int64),
            "beam_idx": np.zeros((1,), dtype=np.int32),
        })
        next_id = int(np.asarray(req.get_tensor(logits_out).data)[0, -1].argmax())
        past_len += 1
    snap_after = snapshot_state(req)

    by_pre, total_pre = aggregate(snap_prefill)
    by_post, total_post = aggregate(snap_after)

    print(f"\n  Prompt={T} tokens, decoded {past_len - T} tokens "
          f"(total context = {past_len} tokens)")
    print(f"\n  {'category':<22s} {'count':>6s}  {'after prefill':>16s}  {'after decode':>16s}")
    cats = sorted(set(by_pre) | set(by_post))
    for cat in cats:
        cpre, bpre = by_pre.get(cat, [0, 0])
        cpost, bpost = by_post.get(cat, [0, 0])
        print(f"  {cat:<22s} {cpost:>6d}  {fmt_bytes(bpre):>16s}  {fmt_bytes(bpost):>16s}")
    print(f"  {'TOTAL':<22s} {sum(c for c,_ in by_post.values()):>6d}  "
          f"{fmt_bytes(total_pre):>16s}  {fmt_bytes(total_post):>16s}")

    growth = total_post - total_pre
    decoded = past_len - T
    print(f"\n  Cache growth over {decoded} decode steps: {fmt_bytes(growth)} "
          f"= {growth / decoded:.0f} B/token")

    return {
        "T_prompt": T, "T_total": past_len,
        "total_pre": total_pre, "total_post": total_post,
        "by_post": dict(by_post),
    }


def main():
    a = run("UNFUSED (baseline)", apply_fusions=False)
    b = run("FUSED (linear-attn + conv1d + lm_head slice)", apply_fusions=True)

    print(f"\n{'=' * 64}\nSUMMARY: persistent state — unfused vs fused\n{'=' * 64}")
    print(f"{'metric':<40s}  {'unfused':>14s}  {'fused':>14s}")
    print(f"  {'prompt tokens':<38s}  {a['T_prompt']:>14d}  {b['T_prompt']:>14d}")
    print(f"  {'total tokens':<38s}  {a['T_total']:>14d}  {b['T_total']:>14d}")
    print(f"  {'total state after prefill':<38s}  {fmt_bytes(a['total_pre']):>14s}  {fmt_bytes(b['total_pre']):>14s}")
    print(f"  {'total state after generation':<38s}  {fmt_bytes(a['total_post']):>14s}  {fmt_bytes(b['total_post']):>14s}")
    per_tok_a = (a['total_post'] - a['total_pre']) / (a['T_total'] - a['T_prompt'])
    per_tok_b = (b['total_post'] - b['total_pre']) / (b['T_total'] - b['T_prompt'])
    print(f"  {'KV growth per decoded token':<38s}  {per_tok_a:>11.0f} B  {per_tok_b:>11.0f} B")


if __name__ == "__main__":
    main()
