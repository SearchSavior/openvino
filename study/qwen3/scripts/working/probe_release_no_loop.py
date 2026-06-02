"""Validate the hypothesis that the release_memory crash is specific to
TensorIterator (Loop) nodes, not a generic dynamic-shape problem.

Replaces all 18 gated-delta Loops with our v1 GatedDeltaRule custom op.
With no Loops in the graph, TensorIterator's stale-pointer caching bug
shouldn't fire. If release_memory + re-infer works on this model, the
bug is confirmed loop-specific."""
import ctypes, subprocess, sys
from pathlib import Path
import openvino as ov
import numpy as np

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WORK = Path("/tmp/qwen3-work")
SO = Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so"

libc = ctypes.CDLL("libc.so.6")
libc.malloc_trim.argtypes = [ctypes.c_int]
libc.malloc_trim.restype = ctypes.c_int


def rss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024


def prep_no_loop():
    """Rewrite all 18 gated-delta Loops with v1 custom op, serialize."""
    kdir = Path(__file__).resolve().parents[2] / "kernels"
    code = f"""
import sys; sys.path.insert(0, '{kdir}')
import openvino as ov
from fused_linear_attn import register as rc, replace_gated_delta_rule_loops
from lm_head_slice import slice_lm_head_to_last_token

c = ov.Core(); rc(c)
m = c.read_model('{ORIG}/openvino_language_model.xml')
n_loops_before = sum(1 for op in m.get_ops() if op.get_type_name() == 'Loop')
n_replaced = replace_gated_delta_rule_loops(m)
n_loops_after = sum(1 for op in m.get_ops() if op.get_type_name() == 'Loop')
slice_lm_head_to_last_token(m)
print(f'loops before={{n_loops_before}}  replaced={{n_replaced}}  loops after={{n_loops_after}}')
ov.serialize(m, '{WORK}/probe_noloop.xml', '{WORK}/probe_noloop.bin')
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        sys.exit("prep failed")
    print(r.stdout.strip())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1024)
    args = ap.parse_args()
    T = args.T

    prep_no_loop()

    print(f"\npre-load:                {rss():7.1f} MiB")
    core = ov.Core()
    core.add_extension(str(SO))
    lm = core.read_model(f"{WORK}/probe_noloop.xml")
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})
    print(f"after compile:           {rss():7.1f} MiB")

    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    hidden = lm.input("inputs_embeds").get_partial_shape()[2].get_length()
    rng = np.random.default_rng(0)
    feeds = lambda: {
        "inputs_embeds":  ov.Tensor(rng.standard_normal((1, T, hidden), dtype=np.float32) * 0.01),
        "attention_mask": ov.Tensor(np.ones((1, T), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.tile(np.arange(T, dtype=np.int64).reshape(1, 1, T), (pid_b, 1, 1))),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }

    req = compiled.create_infer_request()
    req.infer(feeds())
    print(f"after warmup (no Loop):  {rss():7.1f} MiB")

    del req
    compiled.release_memory()
    libc.malloc_trim(0)
    print(f"after release+trim:      {rss():7.1f} MiB")

    print("\nTrying re-infer (the key test):")
    try:
        req = compiled.create_infer_request()
        req.infer(feeds())
        print(f"  SUCCESS — re-infer worked.  RSS = {rss():7.1f} MiB")
        print("  => bug IS loop-specific. TensorIterator stale-pointer caching is the cause.")
    except RuntimeError as e:
        msg = str(e).split("\n")[0][:120]
        print(f"  CRASHED — {msg}")
        print("  => bug is broader than TensorIterator. Need to look elsewhere.")


if __name__ == "__main__":
    main()
