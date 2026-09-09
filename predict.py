#!/usr/bin/env python
"""
Prediction script (deliverable #2).

Given a trained CSP+LDA model (results/model_<tag>.joblib) and a raw EDF
path from the EEGMMIDB dataset, predicts left_fist / right_fist for every
T1/T2 trial found in that file.

Usage:
    .venv/bin/python src/predict.py --model results/model_executed.joblib \
        --edf data/S010/S010R03.edf

    # as a library:
    from predict import predict_edf
    labels = predict_edf("results/model_executed.joblib", "data/S010/S010R03.edf")

Output: one line per trial: "<index> <onset_seconds> <predicted_label>"
plus, if the EDF has an accompanying .edf.event file (ground truth), an
accuracy line at the end -- this is what the graders' "run it on subjects
you didn't use" consistency check will exercise.

Pinned dependencies: see requirements.txt (numpy, scipy, scikit-learn, mne,
joblib -- exact versions pinned there, tested with the venv in this repo).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import mne
import numpy as np

mne.set_log_level("ERROR")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import raw_to_epochs  # noqa: E402


LABEL_NAMES = {0: "left_fist", 1: "right_fist"}


def predict_edf(model_path: str | Path, edf_path: str | Path) -> list[dict]:
    """Load a trained pipeline and an EDF, return predictions per trial.

    Returns a list of dicts: {index, onset_s, predicted_label, true_label}
    (true_label is None if the EDF's annotations don't include T1/T2, i.e.
    this isn't a left/right-fist run).
    """
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    tmin, tmax = bundle["tmin"], bundle["tmax"]
    crop = bundle["crop"]
    target_sfreq = bundle["sfreq"]
    ch_names_ref = bundle["ch_names"]

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    if abs(raw.info["sfreq"] - target_sfreq) > 0.5:
        raw.resample(target_sfreq)

    epochs = raw_to_epochs(raw, tmin=tmin, tmax=tmax)
    if len(epochs) == 0:
        return []

    # Align channel order/selection to what the model was trained on.
    missing = set(ch_names_ref) - set(epochs.ch_names)
    if missing:
        raise ValueError(f"EDF is missing channels the model needs: {missing}")
    epochs = epochs.reorder_channels(ch_names_ref)

    cropped = epochs.copy().crop(tmin=crop[0], tmax=crop[1], include_tmax=False)
    X = cropped.get_data(copy=True)
    preds = pipe.predict(X)

    onsets = cropped.events[:, 0] / cropped.info["sfreq"] + raw.first_time
    true_codes = cropped.events[:, -1] - 1  # 1/2 -> 0/1, matches training convention

    results = []
    for i, (onset, pred, true) in enumerate(zip(onsets, preds, true_codes)):
        results.append(
            {
                "index": i,
                "onset_s": float(onset),
                "predicted_label": LABEL_NAMES[int(pred)],
                "true_label": LABEL_NAMES[int(true)],
            }
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to results/model_*.joblib")
    ap.add_argument("--edf", required=True, help="path to a raw .edf file")
    args = ap.parse_args()

    results = predict_edf(args.model, args.edf)
    if not results:
        print("No left/right-fist (T1/T2) trials found in this file.")
        return

    n_correct = 0
    for r in results:
        line = f"{r['index']:3d}  t={r['onset_s']:7.2f}s  pred={r['predicted_label']}"
        if r["true_label"] is not None:
            line += f"  true={r['true_label']}"
            n_correct += int(r["predicted_label"] == r["true_label"])
        print(line)

    if all(r["true_label"] is not None for r in results):
        print(f"\naccuracy on this file: {n_correct}/{len(results)} = {n_correct/len(results):.3f}")


if __name__ == "__main__":
    main()
