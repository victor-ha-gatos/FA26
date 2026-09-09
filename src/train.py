"""
Main experiment driver. Builds datasets, runs the three evaluation regimes
from evaluate.py, and writes a results summary + a couple of plots.

Usage:
    .venv/bin/python src/train.py --subjects 1-20 --run_type executed
    .venv/bin/python src/train.py --subjects 1-109 --run_type imagined --loso_only

Kept deliberately simple / script-like (not argparse-heavy) since this is
an exploratory research task, not a shipped tool -- see predict.py for the
deliverable-grade CLI.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data import build_dataset, BAD_SUBJECTS
from model import build_pipeline
from evaluate import (
    within_subject_cv,
    leave_one_subject_out,
    group_kfold_cv,
    subject_split_cv,
    summarize,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_subjects(spec: str) -> list[int]:
    subs = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            subs.extend(range(int(lo), int(hi) + 1))
        else:
            subs.append(int(part))
    return sorted(set(subs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-20", help="e.g. 1-20 or 1,2,5-10")
    ap.add_argument("--run_type", default="executed", choices=["executed", "imagined", "both"])
    ap.add_argument("--n_components", type=int, default=4)
    ap.add_argument("--loso_only", action="store_true", help="skip within-subject CV (slow for many subjects)")
    ap.add_argument("--skip_permutation", action="store_true")
    ap.add_argument("--tag", default=None, help="label for output files; defaults to run_type")
    ap.add_argument(
        "--exclude_bad",
        action="store_true",
        help="exclude known bad subjects (88, 89, 92, 100, 104 -- see data.py BAD_SUBJECTS)",
    )
    ap.add_argument(
        "--exclude_subjects",
        default=None,
        help="comma-separated subject IDs to exclude, e.g. 88,92,100 (in addition to --exclude_bad)",
    )
    ap.add_argument(
        "--train_ratios",
        default=None,
        help="comma-separated subject-level train fractions to test, e.g. 0.5,0.7,0.8,0.9 "
        "(runs subject_split_cv at each ratio; skipped if not given)",
    )
    args = ap.parse_args()

    subjects = parse_subjects(args.subjects)
    exclude = set(BAD_SUBJECTS) if args.exclude_bad else set()
    if args.exclude_subjects:
        exclude |= {int(s) for s in args.exclude_subjects.split(",")}
    tag = args.tag or args.run_type
    print(f"Loading {len(subjects)} subjects, run_type={args.run_type} ...")
    if exclude:
        print(f"Excluding subjects: {sorted(exclude)}")
    t0 = time.time()
    ds = build_dataset(subjects, run_type=args.run_type, exclude_subjects=exclude)
    print(f"Loaded X={ds.X.shape} y={ds.y.shape} in {time.time()-t0:.1f}s")
    print(f"Class balance: left={np.sum(ds.y==0)} right={np.sum(ds.y==1)}")

    def pipeline_fn():
        return build_pipeline(n_components=args.n_components)

    out = {"config": vars(args), "n_trials": int(len(ds.y)), "n_subjects": len(subjects)}

    if not args.loso_only:
        print("\nRunning within-subject CV ...")
        wsc = within_subject_cv(pipeline_fn, ds.X, ds.y, ds.groups)
        print(summarize(wsc, "within-subject CV"))
        out["within_subject_cv"] = wsc

    print("\nRunning leave-one-subject-out CV ...")
    loso = leave_one_subject_out(pipeline_fn, ds.X, ds.y, ds.groups)
    print(summarize(loso, "LOSO (real labels)"))
    out["loso"] = loso

    if not args.skip_permutation:
        print("\nRunning LOSO with permuted labels (baseline) ...")
        loso_perm = leave_one_subject_out(pipeline_fn, ds.X, ds.y, ds.groups, shuffle_labels=True)
        print(summarize(loso_perm, "LOSO (permuted labels, baseline)"))
        out["loso_permuted"] = loso_perm

    if args.train_ratios:
        ratio_results = {}
        for ratio_str in args.train_ratios.split(","):
            ratio = float(ratio_str)
            print(f"\nRunning subject-level train/test split, train_size={ratio} ...")
            res = subject_split_cv(pipeline_fn, ds.X, ds.y, ds.groups, train_size=ratio)
            print(
                f"  train_size={ratio}: mean={res['mean']:.3f} std={res['std']:.3f} "
                f"(over {len(res['folds'])} folds)"
            )
            ratio_results[ratio_str] = res
        out["train_ratio_sweep"] = ratio_results

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"results_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    # Fit one final pipeline on ALL loaded trials and persist it -- this is
    # the artifact predict.py loads. Its LOSO number above (not any
    # within-subject number) is the honest estimate of how this saved model
    # will do on a subject it has never seen, which is exactly what
    # predict.py will be run against.
    import joblib

    final_pipe = pipeline_fn()
    final_pipe.fit(ds.X, ds.y)
    model_path = RESULTS_DIR / f"model_{tag}.joblib"
    joblib.dump(
        {
            "pipeline": final_pipe,
            "ch_names": ds.ch_names,
            "sfreq": ds.sfreq,
            "tmin": ds.tmin,
            "tmax": ds.tmax,
            "crop": ds.crop,
            "run_type": args.run_type,
            "n_components": args.n_components,
        },
        model_path,
    )
    print(f"Wrote {model_path}")


if __name__ == "__main__":
    main()
