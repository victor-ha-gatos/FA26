"""Generates results/loso_accuracy.png: per-subject LOSO accuracy (real vs.
permuted baseline) for both run types, so the spread (not just the mean) is
visible -- the prompt explicitly asks to "show us the range, not the best
case."
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, tag, title in zip(axes, ["executed_40", "imagined_40"], ["Executed (runs 3,7,11)", "Imagined (runs 4,8,12)"]):
    d = json.loads((RESULTS_DIR / f"results_{tag}.json").read_text())
    real = d["loso"]["per_subject"]
    perm = d["loso_permuted"]["per_subject"]
    subs = sorted(int(k) for k in real)
    r = np.array([real[str(s)] for s in subs])
    p = np.array([perm[str(s)] for s in subs])
    order = np.argsort(r)
    ax.scatter(range(len(subs)), r[order], label="real labels", s=18, color="tab:blue")
    ax.scatter(range(len(subs)), p[order], label="permuted labels", s=18, color="tab:gray", alpha=0.7)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, label="chance")
    ax.axhline(r.mean(), color="tab:blue", linestyle=":", linewidth=1.5, label=f"mean real={r.mean():.3f}")
    ax.set_title(title)
    ax.set_xlabel("subjects, sorted by real-label accuracy")
    ax.set_ylim(0.2, 1.0)

axes[0].set_ylabel("LOSO accuracy (left vs. right fist)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("CSP+LDA, leave-one-subject-out, n=40 subjects")
fig.tight_layout()
out = RESULTS_DIR / "loso_accuracy.png"
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
