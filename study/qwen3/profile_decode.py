"""
Profile a decode step of an OpenVINO-exported causal LM after warming a KV cache.

Usage:
    python profile_decode.py --model <ir-dir> [--prefill-len 128] [--decode-warmup 4] [--device CPU]

Outputs (in --output-dir, default ./decode_out):
    exec_graph.xml          Post-fusion execution graph (open in Netron)
    perf_counters.csv       Per-node real_time / cpu_time / kernel
    summary.txt             Sorted breakdowns by op type and kernel type

The measured profile reflects the LAST inference call. The script does a prefill
plus a few decode warmup steps before the measured decode step so cache and JIT'd
kernels are hot.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import openvino as ov

TOP_N = 25


def build_prefill_inputs(model: ov.CompiledModel, prompt_len: int) -> dict:
    inputs = {}
    for inp in model.inputs:
        name = inp.get_any_name()
        if name == "input_ids":
            inputs[name] = np.random.randint(0, 1000, size=(1, prompt_len), dtype=np.int64)
        elif name == "attention_mask":
            inputs[name] = np.ones((1, prompt_len), dtype=np.int64)
        elif name == "position_ids":
            inputs[name] = np.arange(prompt_len, dtype=np.int64).reshape(1, prompt_len)
        elif name == "beam_idx":
            inputs[name] = np.zeros((1,), dtype=np.int32)
        else:
            raise RuntimeError(f"unhandled input: {name} {inp.get_partial_shape()}")
    return inputs


def build_decode_inputs(model: ov.CompiledModel, past_len: int) -> dict:
    inputs = {}
    for inp in model.inputs:
        name = inp.get_any_name()
        if name == "input_ids":
            inputs[name] = np.random.randint(0, 1000, size=(1, 1), dtype=np.int64)
        elif name == "attention_mask":
            inputs[name] = np.ones((1, past_len + 1), dtype=np.int64)
        elif name == "position_ids":
            inputs[name] = np.array([[past_len]], dtype=np.int64)
        elif name == "beam_idx":
            inputs[name] = np.zeros((1,), dtype=np.int32)
        else:
            raise RuntimeError(f"unhandled input: {name} {inp.get_partial_shape()}")
    return inputs


def summarize(prof, total_us: int, past_len: int, out_path: Path) -> None:
    by_kernel = defaultdict(lambda: [0, 0])
    by_op = defaultdict(lambda: [0, 0])
    executed = []
    for p in prof:
        if not str(p.status).endswith("EXECUTED"):
            continue
        us = p.real_time.microseconds
        by_kernel[p.exec_type][0] += us; by_kernel[p.exec_type][1] += 1
        by_op[p.node_type][0] += us; by_op[p.node_type][1] += 1
        executed.append(p)

    with out_path.open("w") as f:
        f.write(f"Decode profile — past_len={past_len} (single token, KV cache warm)\n")
        f.write(f"Total executed time: {total_us} us ({total_us/1000:.1f} ms)\n\n")

        f.write("=== By op type (node_type) ===\n")
        for op, (us, n) in sorted(by_op.items(), key=lambda x: -x[1][0])[:TOP_N]:
            f.write(f"  {us:>10} us  {n:>4} calls  {us*100/total_us:5.1f}%  {op}\n")

        f.write("\n=== By kernel (exec_type) ===\n")
        for k, (us, n) in sorted(by_kernel.items(), key=lambda x: -x[1][0])[:TOP_N]:
            f.write(f"  {us:>10} us  {n:>4} calls  {us*100/total_us:5.1f}%  {k}\n")

        f.write(f"\n=== Top {TOP_N} individual nodes ===\n")
        for p in sorted(executed, key=lambda p: -p.real_time.microseconds)[:TOP_N]:
            f.write(f"  {p.real_time.microseconds:>10} us  "
                    f"{p.node_type:35s}  {p.exec_type:30s}  {p.node_name}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="IR directory containing openvino_model.xml")
    ap.add_argument("--prefill-len", type=int, default=128, help="prompt length to warm KV cache")
    ap.add_argument("--decode-warmup", type=int, default=4, help="decode steps before the measured one")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--threads", type=int, default=0, help="0 = let plugin decide")
    ap.add_argument("--output-dir", default="./decode_out")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core = ov.Core()
    model = core.read_model(str(Path(args.model) / "openvino_model.xml"))

    config = {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True}
    if args.threads:
        config["INFERENCE_NUM_THREADS"] = args.threads
    compiled = core.compile_model(model, args.device, config)

    ov.serialize(compiled.get_runtime_model(), str(out_dir / "exec_graph.xml"))

    req = compiled.create_infer_request()

    # Populate KV cache (untimed).
    req.infer(build_prefill_inputs(compiled, args.prefill_len))
    past_len = args.prefill_len

    # Warmup decodes so JIT'd kernels are hot and cache is warm.
    for _ in range(args.decode_warmup):
        req.infer(build_decode_inputs(compiled, past_len))
        past_len += 1

    # Measured decode — get_profiling_info() reflects this call.
    req.infer(build_decode_inputs(compiled, past_len))
    prof = req.get_profiling_info()

    csv_path = out_dir / "perf_counters.csv"
    total_us = 0
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_name", "status", "node_type", "exec_type", "real_time_us", "cpu_time_us"])
        for p in prof:
            w.writerow([p.node_name, str(p.status).split('.')[-1], p.node_type, p.exec_type,
                        p.real_time.microseconds, p.cpu_time.microseconds])
            if str(p.status).endswith("EXECUTED"):
                total_us += p.real_time.microseconds

    summary_path = out_dir / "summary.txt"
    summarize(prof, total_us, past_len, summary_path)

    print(summary_path.read_text())
    print(f"\nArtifacts in {out_dir}/")


if __name__ == "__main__":
    main()
