"""Figure: empirical CDF of principal angles between added-capacity subspaces.

Reads results/diagnose_rank_collapse/cdf_angles.json (produced by the rank-gap
diagnostic) and draws the empirical CDF of the K=36 principal angles for:
  - B-vs-C   : domain-B-specialized vs domain-C-specialized gate subspaces
  - grown->CL: grown (at init) vs after-CL gate subspace
  - null     : two independent random 36-dim subspaces in R^D
against a random-subspace null that concentrates near 90 degrees. The B-vs-C
curve hugging the null (with a small small-angle tail) is the re-tasking result
(E6): dimension conserved, directions rotated to near-orthogonal.

Has an ``if __name__ == '__main__'`` guard so importing it does nothing.
Run:  python experiments/E6_subspace_rotation/plot_subspace_rotation.py
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "results/diagnose_rank_collapse/cdf_angles.json"
# Written to the repo figures/ directory.
OUT_PATHS = [
    "figures/subspace_rotation_cdf.pdf",
]


def ecdf(values):
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def main():
    with open(DATA) as f:
        d = json.load(f)

    series = [
        ("null", d["null"], "#9e9e9e", "--", "null (random subspaces)"),
        ("b_vs_c", d["b_vs_c"], "#1f77b4", "-", "B-specialized vs C-specialized"),
        ("grown_cl", d["grown_cl"], "#2ca02c", "-.", r"grown $\to$ after-CL"),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for _key, vals, color, ls, label in series:
        x, y = ecdf(vals)
        ax.plot(x, y, color=color, linestyle=ls, linewidth=2.0, label=label)

    # ~5/36 shared-core reference (directions below 80 degrees).
    ax.axvline(80.0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.text(79.0, 0.06, "shared core\n($\\approx$5/36 below $80^\\circ$)",
            ha="right", va="bottom", fontsize=8, color="#555555")

    ax.set_xlim(45, 90)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Principal angle (degrees)")
    ax.set_ylabel("Cumulative fraction of the 36 directions")
    ax.grid(True, which="major", color="#eeeeee", linewidth=0.8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()

    for out in OUT_PATHS:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
