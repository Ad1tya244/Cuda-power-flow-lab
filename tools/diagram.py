#!/usr/bin/env python3
"""diagram.py - draws the GPU Newton-Raphson data-path figure for the report (matplotlib)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT, exist_ok=True)

INK, QUIET, EDGE = "#1a1a1a", "#505050", "#8a8a8a"
LIB, OWN, HOST = "#dbe8f7", "#dff0df", "#f2f2f2"

fig, ax = plt.subplots(figsize=(9.6, 6.3), dpi=200)
ax.set_xlim(0, 960)
ax.set_ylim(0, 630)
ax.axis("off")


def box(x, y, w, h, title, lines, fill, strong=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10", facecolor=fill,
                                edgecolor=INK if strong else EDGE, linewidth=1.5 if strong else 1))
    ax.text(x + 12, y + h - 18, title, fontsize=9.5, fontweight="bold", color=INK, va="center")
    for k, ln in enumerate(lines):
        ax.text(x + 12, y + h - 38 - 16 * k, ln, fontsize=8, color=QUIET, va="center")


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=EDGE, lw=1.1))


ax.text(20, 612, "One Newton-Raphson load-flow solve on the GPU", fontsize=12, fontweight="bold", color=INK, va="center")

# ---- host side
box(20, 470, 250, 92, "Host (CPU) prepares", ["Ybus: sparse, CSR, complex", "Jacobian pattern, bus colours",
                                               "specified P and Q, start voltages"], HOST)
ax.text(20, 452, "Steps 1-2: cudaMalloc, cudaMemcpy H->D", fontsize=8, color=QUIET, va="center")
arrow((270, 516), (330, 516))
box(20, 44, 250, 80, "Host (CPU) receives", ["magnitude and angle of the", "voltage at every bus"], HOST)
ax.text(20, 140, "Steps 4-5: cudaMemcpy D->H, cudaFree", fontsize=8, color=QUIET, va="center")
arrow((330, 84), (270, 84))

# legend + note
ax.add_patch(FancyBboxPatch((20, 318), 18, 12, boxstyle="round,pad=0,rounding_size=2", facecolor=OWN, edgecolor=EDGE))
ax.text(46, 324, "our CUDA kernels", fontsize=8, color=QUIET, va="center")
ax.add_patch(FancyBboxPatch((20, 294), 18, 12, boxstyle="round,pad=0,rounding_size=2", facecolor=LIB, edgecolor=EDGE))
ax.text(46, 300, "cuSPARSE / cuBLAS library + our kernels", fontsize=8, color=QUIET, va="center")
ax.text(20, 256, "Only the voltages, and one number per\nBiCGSTAB iteration, cross the PCIe bus.",
        fontsize=8, color=QUIET, va="center")

# ---- GPU Newton loop
ax.add_patch(FancyBboxPatch((330, 24), 610, 568, boxstyle="round,pad=0,rounding_size=12", facecolor="#fbfbfb",
                            edgecolor=EDGE, linewidth=1))
ax.text(345, 572, "Step 3: Newton loop on the GPU (4-5 iterations)", fontsize=10, fontweight="bold", color=INK,
        va="center")
X, W = 345, 520
box(X, 436, W, 112, "1  Currents and mismatch", ["k_voltage: V = Vm e^(jVa), one thread per bus",
                                                  "cuSPARSE SpMV: I = Ybus V (complex, only non-zeros)",
                                                  "k_mismatch: F = V conj(I) - S_spec, one thread per equation",
                                                  "stop when max|F| < 1e-8 pu"], OWN)
box(X, 354, W, 64, "2  Jacobian values", ["k_jacobian: one thread per NON-ZERO of the Jacobian",
                                          "zeros are never stored or visited"], OWN)
box(X, 272, W, 64, "3  ILU(0) preconditioner", ["cusparseDcsrilu02 on the Jacobian's own sparsity pattern",
                                                "buses coloured so linked buses never share a colour"], LIB)
box(X, 126, W, 128, "4  BiCGSTAB linear solve  J dx = -F", ["2 x cuSPARSE SpMV (J x) per iteration",
                                                              "4 x triangular solves with our colour kernels:",
                                                              "   one thread per bus of the current colour",
                                                              "cuBLAS dot products, scalars kept on the GPU",
                                                              "tolerance 0.1 x max|F| (inexact Newton)"], LIB, strong=True)
box(X, 44, W, 64, "5  Update", ["k_update: angles Va += dx, load-bus magnitudes Vm += dx",
                                "then back to step 1"], OWN)
for y1, y2 in [(436, 418), (354, 336), (272, 254), (126, 108)]:
    arrow((X + W / 2, y1), (X + W / 2, y2))
# loop-back path on the right
ax.plot([865, 905, 905], [76, 76, 492], color=EDGE, lw=1.1)
arrow((905, 492), (866, 492))
ax.text(918, 284, "next Newton step", fontsize=8, color=QUIET, rotation=90, va="center", ha="center")

fig.savefig(os.path.join(OUT, "fig_datapath.png"), bbox_inches="tight", facecolor="white")
print("wrote results/figures/fig_datapath.png")
