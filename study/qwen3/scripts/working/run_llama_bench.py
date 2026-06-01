#!/usr/bin/env python3
"""Spawn llama-bench at a given prompt size and poll its /proc/<pid>/status
every 50 ms, recording timeline of VmRSS + VmHWM. Also captures /usr/bin/time
peak RSS. Reports trajectory."""
import argparse
import os
import re
import subprocess
import time
from pathlib import Path

LLAMA_BIN = "/tmp/llama.cpp/build/bin/llama-bench"
MODEL = "/tmp/qwen35-0.8b-Q8_0.gguf"


def read_status(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            d = {}
            for line in f:
                if ":" not in line: continue
                k, _, rest = line.partition(":")
                d[k.strip()] = rest.strip()
        return d
    except FileNotFoundError:
        return None


def to_mib(s):
    m = re.match(r"(\d+)\s*kB", s)
    if not m: return 0.0
    return int(m.group(1)) / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=8192, help="prompt length")
    ap.add_argument("--n", type=int, default=32, help="tokens to generate")
    ap.add_argument("--t", type=int, default=4, help="threads")
    ap.add_argument("--interval", type=float, default=0.05)
    args = ap.parse_args()

    cmd = [LLAMA_BIN, "-m", MODEL,
           "-p", str(args.p), "-n", str(args.n),
           "-t", str(args.t), "-r", "1"]
    print(f"$ {' '.join(cmd)}")

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t0 = time.time()
    trajectory = []
    while p.poll() is None:
        st = read_status(p.pid)
        if st is None: break
        rss = to_mib(st.get("VmRSS", "0 kB"))
        hwm = to_mib(st.get("VmHWM", "0 kB"))
        trajectory.append((time.time() - t0, rss, hwm))
        time.sleep(args.interval)
    stdout, stderr = p.communicate()
    elapsed = time.time() - t0

    # llama-bench printed throughput to stdout
    print(stdout.decode(errors="replace"))

    if trajectory:
        peak_kb = int(max(h for _, _, h in trajectory) * 1024)
        print(f"\npeak VmHWM observed via /proc poll = {peak_kb/1024:.1f} MiB")
    else:
        peak_kb = None

    # Trajectory summary
    if not trajectory:
        print("(no samples)"); return
    print(f"\nTrajectory (sampled every {args.interval*1000:.0f} ms, {len(trajectory)} samples, wall {elapsed:.1f}s):")
    print(f"{'t (s)':>8s}  {'VmRSS (MiB)':>12s}  {'VmHWM (MiB)':>12s}")
    # Print every Nth sample to keep output bounded
    stride = max(1, len(trajectory) // 60)
    for i in range(0, len(trajectory), stride):
        t, r, h = trajectory[i]
        print(f"{t:>8.2f}  {r:>12.1f}  {h:>12.1f}")
    t, r, h = trajectory[-1]
    print(f"{t:>8.2f}  {r:>12.1f}  {h:>12.1f}   (final)")
    max_rss = max(r for _, r, _ in trajectory)
    max_hwm = max(h for _, _, h in trajectory)
    print(f"\nMax observed in-process VmRSS = {max_rss:.1f} MiB")
    print(f"Max observed in-process VmHWM = {max_hwm:.1f} MiB")


if __name__ == "__main__":
    main()
