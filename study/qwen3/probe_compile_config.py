"""
Find which compile_model property makes vlm-unfused half the size of raw unfused.

VLMPipeline at seq=1024 peaks at 2221 MB for unfused while raw ov.Core peaks
at 4359 MB. Same .xml, same .bin, same .so. The only difference is the
compile properties dictionary passed by genai vs by us.

Probes a sequence of {prop: value} combinations and reports peak RSS for
each, all in clean subprocesses against the unfused IR.
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIG_XML = "/tmp/qwen3-work/qwen35-0.8b-int8/openvino_language_model.xml"
HIDDEN = 1024
SEQ = 1024


def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])


def worker(props_json):
    import numpy as np, openvino as ov
    props = json.loads(props_json)
    core = ov.Core()
    model = core.read_model(ORIG_XML)
    compiled = core.compile_model(model, "CPU", props)
    req = compiled.create_infer_request()

    rng = np.random.default_rng(0)
    inp = {
        "inputs_embeds": (rng.standard_normal((1, SEQ, HIDDEN)) * 0.02).astype(np.float32),
        "attention_mask": np.ones((1, SEQ), dtype=np.int64),
        "position_ids": np.tile(np.arange(SEQ, dtype=np.int64).reshape(1, 1, SEQ), (4, 1, 1)),
        "beam_idx": np.zeros((1,), dtype=np.int32),
    }

    sampled = []
    stop = threading.Event()
    def sampler():
        while not stop.is_set():
            sampled.append(rss_kb()); time.sleep(0.005)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    t0 = time.time()
    req.infer(inp)
    elapsed = time.time() - t0
    stop.set(); th.join()
    peak = max(sampled) / 1024
    end = rss_kb() / 1024
    print("@@RESULT@@", json.dumps({
        "props": props, "peak_mb": peak, "end_mb": end, "infer_s": elapsed
    }))


def run_one(props):
    cmd = [sys.executable, str(__file__), "--worker", json.dumps(props)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout); print(res.stderr)
        return None
    line = next(L for L in res.stdout.splitlines() if L.startswith("@@RESULT@@"))
    return json.loads(line[len("@@RESULT@@ "):])


CONFIGS = [
    ("baseline (THREADS=4 only)", {"INFERENCE_NUM_THREADS": 4}),
    ("PERF_HINT=LATENCY", {"INFERENCE_NUM_THREADS": 4, "PERFORMANCE_HINT": "LATENCY"}),
    ("PERF_HINT=THROUGHPUT", {"INFERENCE_NUM_THREADS": 4, "PERFORMANCE_HINT": "THROUGHPUT"}),
    ("KV_CACHE_PRECISION=u8", {"INFERENCE_NUM_THREADS": 4, "KV_CACHE_PRECISION": "u8"}),
    ("DYNAMIC_QG=32", {"INFERENCE_NUM_THREADS": 4, "DYNAMIC_QUANTIZATION_GROUP_SIZE": 32}),
    ("INFERENCE_PRECISION=bf16", {"INFERENCE_NUM_THREADS": 4, "INFERENCE_PRECISION_HINT": "bf16"}),
    ("INFERENCE_PRECISION=f16", {"INFERENCE_NUM_THREADS": 4, "INFERENCE_PRECISION_HINT": "f16"}),
    ("KV_CACHE=u8 + DYNAMIC_QG=32",
     {"INFERENCE_NUM_THREADS": 4, "KV_CACHE_PRECISION": "u8", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 32}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", help=argparse.SUPPRESS, nargs="?")
    args, rest = ap.parse_known_args()
    if args.worker is not None:
        worker(args.worker)
        return

    print(f"unfused IR, seq={SEQ}, INFERENCE_NUM_THREADS=4 throughout")
    print(f"\n{'label':<36s} {'peak(MB)':>10s} {'end(MB)':>10s} {'infer(s)':>10s}")
    print("-" * 72)
    for label, props in CONFIGS:
        r = run_one(props)
        if r is None:
            print(f"{label:<36s} FAILED"); continue
        print(f"{label:<36s} {r['peak_mb']:>10.0f} {r['end_mb']:>10.0f} {r['infer_s']:>10.2f}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        # bypass argparse: positional args
        idx = sys.argv.index("--worker")
        worker(sys.argv[idx + 1])
    else:
        main()
