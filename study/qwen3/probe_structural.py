"""
Test structural hypotheses for why VLMPipeline's unfused peak is half of
raw ov.Core's peak at seq=1024 (2221 vs 4358 MB).

Probes (all against the unfused IR, in clean subprocesses):
  (1) baseline:        dynamic shapes, single infer(seq=1024)
  (2) static-reshape:  model.reshape({inputs_embeds: [1,1024,1024], ...}) then compile
  (3) chunked-1x256:   single infer of seq=1024 — control
  (4) chunked-4x256:   four sequential infer calls of seq=256 each (stateful KV)
  (5) chunked-8x128:   eight sequential infer calls of seq=128 each
  (6) chunked-16x64:   sixteen sequential infer calls of seq=64 each

Each chunk run feeds attention_mask of cumulative length and position_ids
incrementing. State (linear_attn + conv1d + full_attn KV) persists across
infer() calls thanks to ReadValue/Assign, so chunked prefill yields the same
final state as single-shot — just at different memory footprints.
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

ORIG_XML = "/tmp/qwen3-work/qwen35-0.8b-int8/openvino_language_model.xml"
FUSED_XML = "/tmp/qwen3-work/qwen35-0.8b-int8-fused/openvino_language_model.xml"
SO_PATH = Path(__file__).resolve().parent / "cpp_ext/build/libqwen3_ov_ext.so"
HIDDEN = 1024
TOTAL_SEQ = 1024


def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])


def worker(mode):
    import openvino as ov
    core = ov.Core()
    if mode.startswith("fused-"):
        core.add_extension(str(SO_PATH))
        model = core.read_model(FUSED_XML)
    else:
        model = core.read_model(ORIG_XML)

    if mode == "static-reshape":
        # All inputs reshaped to seq=1024 (fixed). attention_mask grows so we use the
        # max past_len too.
        model.reshape({
            "inputs_embeds": ov.PartialShape([1, TOTAL_SEQ, HIDDEN]),
            "attention_mask": ov.PartialShape([1, TOTAL_SEQ]),
            "position_ids": ov.PartialShape([4, 1, TOTAL_SEQ]),
            "beam_idx": ov.PartialShape([1]),
        })
    compiled = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 4})
    req = compiled.create_infer_request()

    rng = np.random.default_rng(0)
    embeds_full = (rng.standard_normal((1, TOTAL_SEQ, HIDDEN)) * 0.02).astype(np.float32)

    suffix = mode.split("-", 1)[1] if "-" in mode else mode
    if suffix in ("baseline", "static-reshape"):
        chunks = [(0, TOTAL_SEQ)]
    elif suffix == "chunked-4x256":
        chunks = [(i, 256) for i in range(0, TOTAL_SEQ, 256)]
    elif suffix == "chunked-8x128":
        chunks = [(i, 128) for i in range(0, TOTAL_SEQ, 128)]
    elif suffix == "chunked-16x64":
        chunks = [(i, 64) for i in range(0, TOTAL_SEQ, 64)]
    else:
        raise ValueError(mode)

    sampled = []
    stop = threading.Event()
    def sampler():
        while not stop.is_set():
            sampled.append(rss_kb()); time.sleep(0.005)
    th = threading.Thread(target=sampler, daemon=True); th.start()

    t0 = time.time()
    for start, length in chunks:
        past = start
        embeds = embeds_full[:, start:start + length, :]
        attn = np.ones((1, past + length), dtype=np.int64)
        pos = np.tile(np.arange(start, start + length, dtype=np.int64).reshape(1, 1, length),
                       (4, 1, 1))
        req.infer({
            "inputs_embeds": embeds,
            "attention_mask": attn,
            "position_ids": pos,
            "beam_idx": np.zeros((1,), dtype=np.int32),
        })
    elapsed = time.time() - t0
    stop.set(); th.join()
    peak = max(sampled) / 1024
    end = rss_kb() / 1024
    print("@@RESULT@@", json.dumps({
        "mode": mode, "chunks": len(chunks), "peak_mb": peak,
        "end_mb": end, "infer_s": elapsed,
    }))


def main():
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        worker(sys.argv[i + 1])
        return

    modes = [
        "unfused-baseline",
        "unfused-chunked-4x256",
        "unfused-chunked-8x128",
        "unfused-chunked-16x64",
        "fused-baseline",
        "fused-chunked-4x256",
        "fused-chunked-8x128",
        "fused-chunked-16x64",
    ]
    print(f"unfused IR @ total_seq={TOTAL_SEQ}, INFERENCE_NUM_THREADS=4")
    print(f"\n{'mode':<22s} {'chunks':>7s} {'peak(MB)':>10s} {'end(MB)':>10s} {'infer(s)':>10s}")
    print("-" * 64)
    for m in modes:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", m]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"{m:<22s} FAILED")
            print(res.stdout); print(res.stderr); continue
        line = next(L for L in res.stdout.splitlines() if L.startswith("@@RESULT@@"))
        r = json.loads(line[len("@@RESULT@@ "):])
        print(f"{r['mode']:<22s} {r['chunks']:>7d} {r['peak_mb']:>10.0f} "
              f"{r['end_mb']:>10.0f} {r['infer_s']:>10.2f}")


if __name__ == "__main__":
    main()
