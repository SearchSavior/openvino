"""
Profile a single prefill pass of an OpenVINO-exported causal LM.

Usage:
    python profile_prefill.py --model <ir-dir> [--prompt-len 512] [--device CPU]

Outputs (in --output-dir, default ./prefill_out):
    exec_graph.xml          Post-fusion execution graph (open in Netron)
    perf_counters.csv       Per-node real_time / cpu_time / kernel
    summary.txt             Sorted breakdowns by op type and kernel type

Tested against optimum-intel-exported Qwen3 / Qwen2.5 / Llama IRs (stateful KV cache).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import openvino as ov

TOP_N = 25


def build_prefill_inputs(model: ov.CompiledModel, prompt_len: int) -> dict:
    """Build a synthetic prefill batch for any stateful causal LM IR.

    Stateful IRs (the default optimum-intel export) expose only:
      input_ids, attention_mask, position_ids, beam_idx
    KV cache lives inside the graph as ReadValue/Assign — no past_key_values inputs.
    """
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


def summarize(prof, total_us: int, prompt_len: int, out_path: Path) -> None:
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
        f.write(f"Prefill profile — prompt_len={prompt_len}\n")
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
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--threads", type=int, default=0, help="0 = let plugin decide")
    ap.add_argument("--output-dir", default="./prefill_out")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core = ov.Core()
    model = core.read_model(str(Path(args.model) / "openvino_model.xml"))

    config = {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True}
    if args.threads:
        config["INFERENCE_NUM_THREADS"] = args.threads
    compiled = core.compile_model(model, args.device, config)

    # Dump post-fusion graph (independent of runtime — does not require an inference).
    ov.serialize(compiled.get_runtime_model(), str(out_dir / "exec_graph.xml"))

    req = compiled.create_infer_request()

    # Warmup discarded. reset_state() clears the in-graph KV buffer so the measured
    # pass starts from past_len=0, same as a real first prefill.
    req.infer(build_prefill_inputs(compiled, args.prompt_len))
    req.reset_state()

    req.infer(build_prefill_inputs(compiled, args.prompt_len))
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
    summarize(prof, total_us, args.prompt_len, summary_path)

    print(summary_path.read_text())
    print(f"\nArtifacts in {out_dir}/")


if __name__ == "__main__":
    main()
