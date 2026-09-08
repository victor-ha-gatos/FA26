"""
Data loading / preprocessing for the PhysioNet EEG Motor Movement/Imagery
Database (EEGMMIDB), scoped to the left-fist vs. right-fist task.

Run semantics (from the dataset docs and confirmed by inspecting annotations):
    Runs 1, 2            : baseline, eyes open / eyes closed (no T1/T2, unused here)
    Runs 3, 7, 11         : executed  left vs. right fist   -> T1 = left fist, T2 = right fist
    Runs 4, 8, 12         : imagined  left vs. right fist   -> T1 = left fist, T2 = right fist
    Runs 5, 9, 13         : executed  both fists vs. both feet (unused here)
    Runs 6, 10, 14        : imagined  both fists vs. both feet (unused here)

T0 in every run marks rest and is dropped for this task -- we only classify
left fist vs. right fist trials.

A handful of subjects (88, 89, 92, 100, 104) were recorded at 128 Hz instead
of 160 Hz, and/or have a different number of samples per trial. We resample
everything to a common rate so channel counts/feature shapes stay consistent
across subjects. See ToDo.md, "Known data-quality issues".
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import mne
import numpy as np

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RUNS_EXECUTED = (3, 7, 11)
RUNS_IMAGINED = (4, 8, 12)

TARGET_SFREQ = 160.0  # common sample rate; a few subjects were recorded at 128 Hz
EVENT_ID = {"left_fist": 1, "right_fist": 2}  # T1 -> left, T2 -> right (see above)

RunType = Literal["executed", "imagined", "both"]


def _runs_for(run_type: RunType) -> tuple[int, ...]:
    if run_type == "executed":
        return RUNS_EXECUTED
    if run_type == "imagined":
        return RUNS_IMAGINED
    if run_type == "both":
        return RUNS_EXECUTED + RUNS_IMAGINED
    raise ValueError(f"unknown run_type {run_type!r}")


def subject_run_path(subject: int, run: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"


def load_raw(subject: int, run: int, data_dir: Path = DATA_DIR) -> mne.io.Raw:
    """Load one EDF run, standardize channel names, resample if needed."""
    path = subject_run_path(subject, run, data_dir)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)  # fixes channel-name quirks (Fc5. -> FC5)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 0.5:
        raw.resample(TARGET_SFREQ)
    return raw


def raw_to_epochs(
    raw: mne.io.Raw,
    tmin: float = -1.0,
    tmax: float = 4.0,
    l_freq: float = 7.0,
    h_freq: float = 30.0,
) -> mne.Epochs:
    """Bandpass-filter (mu+beta, 7-30 Hz) and epoch around T1/T2 events.

    T0 (rest) is intentionally excluded: this project classifies left vs.
    right fist only, not fist-vs-rest.
    """
    raw = raw.copy().filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge")
    events, event_id = mne.events_from_annotations(raw, event_id=dict(T1=1, T2=2))
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"left_fist": 1, "right_fist": 2},
        tmin=tmin,
        tmax=tmax,
        proj=False,
        picks=picks,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    return epochs


@dataclass
class Dataset:
    X: np.ndarray  # (n_trials, n_channels, n_times), cropped window
    y: np.ndarray  # (n_trials,) 0 = left_fist, 1 = right_fist
    groups: np.ndarray  # (n_trials,) subject id, for grouped CV
    run_type: np.ndarray  # (n_trials,) 'executed' or 'imagined', per trial
    ch_names: list[str]
    sfreq: float
    tmin: float
    tmax: float
    crop: tuple[float, float]


def build_dataset(
    subjects: Iterable[int],
    run_type: RunType = "executed",
    data_dir: Path = DATA_DIR,
    tmin: float = -1.0,
    tmax: float = 4.0,
    crop: tuple[float, float] = (1.0, 2.0),
    l_freq: float = 7.0,
    h_freq: float = 30.0,
    verbose: bool = True,
) -> Dataset:
    """Load and epoch every requested run for every subject.

    crop restricts the window used for CSP fitting/classification to the
    steady-state motor imagery/execution period (default 1-2 s post-cue),
    following the standard MNE CSP tutorial. The wider [tmin, tmax] window
    is kept in the returned metadata in case a sliding-window analysis is
    wanted later.
    """
    all_X, all_y, all_groups, all_run_type = [], [], [], []
    ch_names_ref, sfreq_ref = None, None
    runs = _runs_for(run_type)

    for subj in subjects:
        for run in runs:
            path = subject_run_path(subj, run, data_dir)
            if not path.exists():
                if verbose:
                    print(f"  [skip] missing {path}")
                continue
            try:
                raw = load_raw(subj, run, data_dir)
                epochs = raw_to_epochs(raw, tmin=tmin, tmax=tmax, l_freq=l_freq, h_freq=h_freq)
            except Exception as e:  # noqa: BLE001 - log and continue on corrupt files
                print(f"  [error] S{subj:03d}R{run:02d}: {e}")
                continue
            if len(epochs) == 0:
                continue
            cropped = epochs.copy().crop(tmin=crop[0], tmax=crop[1], include_tmax=False)
            X = cropped.get_data(copy=True)
            y = cropped.events[:, -1] - 1  # 0 = left, 1 = right (labels were 1/2)
            if ch_names_ref is None:
                ch_names_ref = cropped.ch_names
                sfreq_ref = cropped.info["sfreq"]
            elif cropped.ch_names != ch_names_ref or abs(cropped.info["sfreq"] - sfreq_ref) > 0.5:
                print(f"  [error] S{subj:03d}R{run:02d}: channel/sfreq mismatch, skipping")
                continue
            all_X.append(X)
            all_y.append(y)
            all_groups.append(np.full(len(y), subj))
            tag = "executed" if run in RUNS_EXECUTED else "imagined"
            all_run_type.append(np.full(len(y), tag, dtype=object))
            if verbose:
                print(f"  S{subj:03d}R{run:02d}: {len(y)} trials")

    if not all_X:
        raise RuntimeError("No data loaded -- check subjects/data_dir")

    return Dataset(
        X=np.concatenate(all_X, axis=0),
        y=np.concatenate(all_y, axis=0),
        groups=np.concatenate(all_groups, axis=0),
        run_type=np.concatenate(all_run_type, axis=0),
        ch_names=ch_names_ref,
        sfreq=sfreq_ref,
        tmin=tmin,
        tmax=tmax,
        crop=crop,
    )
