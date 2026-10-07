#!/usr/bin/env python3
"""plot.py - report figures from the benchmark CSV files in results/ (matplotlib)."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9.5, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})
GPU, CPU1, CPUN, GREY = "#1f6fbf", "#555555", "#aaaaaa", "#888888"


def read(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote results/figures/" + name)


# ---- scaling (final table: GPU vs CPU 1 thread vs CPU 4 threads) -----------
e1 = read("e1_final.csv")
N = np.array([int(r["buses"]) for r in e1])
g = np.array([float(r["gpu_s"]) for r in e1])
c1 = np.array([float(r["cpu1_s"]) for r in e1])
c4 = np.array([float(r["cpu4_s"]) for r in e1])
fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
a.loglog(N, c1, "o-", color=CPU1, label="CPU, 1 thread")
a.loglog(N, c4, "s--", color=CPUN, label="CPU, 4 threads")
a.loglog(N, g, "o-", color=GPU, lw=2.2, label="GPU (GeForce MX250)")
a.set_xlabel("buses in the grid")
a.set_ylabel("full load-flow solve (s, log scale)")
a.set_title("Solve time vs grid size")
a.legend()
best = np.minimum(c1, c4)
sp = best / g
b.semilogx(N, sp, "o-", color=GPU, lw=2.2)
b.axhline(1, color=GREY, lw=1)
b.text(N[0], 1.06, "break-even", fontsize=8, color=GREY)
for x, y in zip(N, sp):
    b.annotate(f"{y:.1f}x", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
b.set_xlabel("buses in the grid")
b.set_ylabel("GPU speed-up over the best CPU run")
b.set_title("The GPU wins from ~20,000 buses up")
b.set_ylim(0, max(sp) * 1.3)
save(fig, "fig_scaling.png")

# ---- phases at 100,000 buses: GPU vs CPU 1 thread -------------------------
e2 = read("e2_phases.csv")
NP = 100000
rows = {r["device"]: r for r in e2 if int(r["buses"]) == NP}
phases = [("inject_ms", "currents + mismatch"), ("jacobian_ms", "Jacobian"), ("ilu_ms", "ILU(0)"),
          ("solve_ms", "BiCGSTAB solve"), ("update_ms", "update")]
fig, ax = plt.subplots(figsize=(9.2, 2.6))
cols = ["#7fb3e6", "#1f6fbf", "#b8b8b8", "#3a3a3a", "#e0a030"]
order = [("gpu", "GPU"), ("cpu1", "CPU 1 thread")]
for k, (dev, lab) in enumerate(order):
    left = 0
    for (key, name), c in zip(phases, cols):
        v = float(rows[dev][key]) / 1000
        ax.barh(k, v, left=left, color=c, label=name if k == 0 else None, height=0.5)
        left += v
    ax.text(left, k, f"  {left:.1f} s", va="center", fontsize=8.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([o[1] for o in order])
ax.invert_yaxis()
ax.set_xlabel("time for the whole solve (s)")
ax.set_title(f"Where the time goes ({NP:,} buses): the linear solve is over 98% on both")
ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32))
ax.grid(axis="y", visible=False)
save(fig, "fig_phases.png")

# ---- convergence ------------------------------------------------------------
e4 = read("e4_convergence.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.4))
sizes = sorted({int(r["buses"]) for r in e4})
shades = np.linspace(0.35, 1.0, len(sizes))
for s, a_ in zip(sizes, shades):
    it = [int(r["iteration"]) for r in e4 if int(r["buses"]) == s]
    mm = [float(r["max_mismatch_pu"]) for r in e4 if int(r["buses"]) == s]
    ax.semilogy(it, mm, "o-", color=GPU, alpha=a_, label=f"{s:,} buses")
ax.axhline(1e-8, color=GREY, lw=1, ls="--")
ax.text(0.1, 2e-8, "tolerance 1e-8 pu", fontsize=8, color=GREY)
ax.set_xlabel("Newton-Raphson iteration")
ax.set_ylabel("largest power mismatch (pu, log)")
ax.set_title("Newton-Raphson converges in 4-5 steps at every size")
ax.legend(fontsize=7.5, ncol=2)
save(fig, "fig_convergence.png")

# ---- sparsity pattern of Ybus (10,000-bus grid) -------------------------------
casef = os.path.join(RES, "synthetic10000.case")
if os.path.exists(casef):
    lines = [ln.split() for ln in open(casef) if ln.strip() and not ln.startswith("#")]
    k = next(i for i, ln in enumerate(lines) if ln[0] == "branches")
    br = np.array([[int(x[0]) - 1, int(x[1]) - 1] for x in lines[k + 1:]])
    n = 10000
    rr = np.concatenate([br[:, 0], br[:, 1], np.arange(n)])
    cc = np.concatenate([br[:, 1], br[:, 0], np.arange(n)])
    nnz = len(set(zip(rr.tolist(), cc.tolist())))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    a.plot(cc, rr, ",", color=GPU)
    a.set_xlim(0, n); a.set_ylim(n, 0)
    a.set_title(f"Ybus, 10,000 x 10,000: {100 * nnz / n / n:.3f}% non-zero")
    a.set_xlabel("bus j"); a.set_ylabel("bus i")
    m = (rr < 120) & (cc < 120)
    b.plot(cc[m], rr[m], "s", color=GPU, ms=2.2)
    b.set_xlim(-1, 120); b.set_ylim(120, -1)
    b.set_title("Zoom on the first 120 buses")
    b.set_xlabel("bus j"); b.set_ylabel("bus i")
    for axx in (a, b):
        axx.grid(False)
    save(fig, "fig_sparsity.png")
print("done")
