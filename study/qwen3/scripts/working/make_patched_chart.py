"""Build the unified chart: patched release_memory vs unpatched floor vs
llama.cpp peak, with prior series labeled."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/user/openvino/study/qwen3/data")


def read(name):
    rows = []
    with open(DATA / name) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


prev = read("nanbeige_with_reset.csv")
new = read("nanbeige_patched_release.csv")
new_by_d = {int(r["depth"]): r for r in new}

depths = [int(r["depth"]) for r in prev]
peaks = [r["ov_peak_rss"] for r in prev]
floors = [r["ov_floor_rss"] for r in prev]
reset_floors = [r["ov_reset_floor_rss"] for r in prev]
llama_peaks = [r["llama_peak_rss"] for r in prev]
patched_floors = [new_by_d[d]["ov_patched_floor_rss"] for d in depths]

fig, ax = plt.subplots(figsize=(11, 7))

ax.plot(depths, peaks,
        "o-", color="tab:red", lw=2.5, ms=8,
        label="OV peak during request (unpatched)")
ax.plot(depths, floors,
        "s--", color="tab:orange", lw=2, ms=8,
        label="OV floor after release_memory (unpatched)")
ax.plot(depths, reset_floors,
        "D:", color="tab:purple", lw=1.5, ms=6, alpha=0.7,
        label="OV floor after reset_state + release_memory (unpatched)")
ax.plot(depths, patched_floors,
        "s-", color="tab:green", lw=2.5, ms=9,
        label="OV floor after release_memory (PATCHED: m_socketWeights.clear)")
ax.plot(depths, llama_peaks,
        "^-", color="black", lw=2.5, ms=9,
        label="llama.cpp peak (Q4_K_M, full request)")

ax.set_xlabel("KV depth (tokens prior to request)", fontsize=11)
ax.set_ylabel("RSS (MiB)", fontsize=11)
ax.set_title("Nanbeige4.1-3B: release_memory floor before/after m_socketWeights.clear patch\n"
             "vs llama.cpp Q4_K_M peak (same model, same hardware)",
             fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=9, framealpha=0.95)
ax.set_xticks(depths)
ax.set_ylim(0, max(peaks) * 1.08)

deltas = [pf - lp for pf, lp in zip(patched_floors, llama_peaks)]
note = "Patched floor vs llama.cpp peak: " + \
       "  ".join(f"d={d}: {delta:+.0f}" for d, delta in zip(depths, deltas))
fig.text(0.5, 0.02, note, ha="center", fontsize=9,
         color="darkgreen", style="italic")

plt.tight_layout(rect=(0, 0.04, 1, 1))
out = DATA / "nanbeige_patched_release.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"Wrote {out}")
