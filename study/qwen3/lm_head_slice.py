"""
lm_head last-token slice rewrite.

The exported LM computes logits for every token in the prefill batch:
    lm_head: [B, T, hidden] @ [hidden, vocab] -> [B, T, vocab]

For next-token generation only the last position's logits are needed. Inserting
a Slice(axis=1, last position only) before the matmul reduces the output from
[B, T, V] to [B, 1, V] — saves T-fold memory on lm_head at prefill (~4 GB at
T=4096, V=248320, FP32) with no semantic change for generation.

For applications that actually need per-position logits (teacher forcing,
pseudo-perplexity), this rewrite would change behavior — but at decode time
T=1 already so it's a no-op there.
"""
from __future__ import annotations

import openvino as ov
from openvino import opset15 as ops


def slice_lm_head_to_last_token(model: ov.Model) -> bool:
    """Insert Slice(axis=1, last position) before the lm_head MatMul.

    Returns True if the rewrite was applied, False if no lm_head was found.
    """
    candidates = [op for op in model.get_ops()
                  if "lm_head" in op.get_friendly_name() and op.get_type_name() == "MatMul"]
    if not candidates:
        return False
    mm = candidates[0]

    act_src = mm.input(0).get_source_output()  # [B, T, hidden]
    # Slice along axis 1 from -1 to INT_MAX with step 1 = take last position only.
    sliced = ops.slice(
        act_src,
        start=ops.constant([-1], dtype=ov.Type.i64),
        stop=ops.constant([2**31 - 1], dtype=ov.Type.i64),
        step=ops.constant([1], dtype=ov.Type.i64),
        axes=ops.constant([1], dtype=ov.Type.i64),
    )
    sliced.set_friendly_name(mm.get_friendly_name() + "/SliceLastToken")
    mm.input(0).replace_source_output(sliced.output(0))
    return True
