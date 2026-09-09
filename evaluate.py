"""
Evaluation harness.

Three evaluation regimes, because the same accuracy number means different
things depending on how train/test were split (this is the "artifact of
measurement" question the assignment asks us to pre-empt):

1. within_subject_cv:
   Trials from the SAME subject appear in both train and test (different
   trials, shuffled). This is the easiest, most optimistic setting -- CSP
   spatial filters and LDA get to fit this exact person's head geometry and
   noise profile. Good for "can this work at all for a given person",
   useless for "will this generalize to a new person".

2. leave_one_subject_out (LOSO):
   Fit CSP+LDA on N-1 subjects, test on the held-out subject, repeat for
   every subject. This is the honest estimate of how a BCI would perform
   on a genuinely new user with no calibration data -- the scenario the
   prompt explicitly cares about ("a new person... almost every design
   decision is downstream of that constraint").

3. permutation baseline:
   Shuffle labels within each subject (preserving the group structure) and
   rerun LOSO. If the shuffled-label accuracy is indistinguishable from the
   real one, the real number is not measuring what we think it's measuring
   (e.g., it could be a subject-identity leak: some subjects' recordings
   are just noisier/cleaner in ways correlated with T1/T2 order or run
   timing, independent of the left/right fist content).

All group-aware splits use GroupKFold / manual LOSO on `groups` (subject
ids) so no subject ever appears in both train and test simultaneously --
this is the single most important methodological control for this dataset,
since within-subject accuracy is inflated by subject-identity confounds
that have nothing to do with left vs right fist.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ShuffleSplit


def within_subject_cv(build_pipeline_fn, X, y, groups, n_splits=5, random_state=42):
    """Per-subject 5-fold CV (shuffled within subject), averaged.

    Returns a dict: subject_id -> mean accuracy, plus overall mean.
    """
    results = {}
    for subj in np.unique(groups):
        mask = groups == subj
        Xs, ys = X[mask], y[mask]
        if len(np.unique(ys)) < 2 or len(ys) < n_splits * 2:
            continue
        cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_state)
        accs = []
        for train_idx, test_idx in cv.split(Xs, ys):
            pipe = clone(build_pipeline_fn())
            pipe.fit(Xs[train_idx], ys[train_idx])
            accs.append(accuracy_score(ys[test_idx], pipe.predict(Xs[test_idx])))
        results[int(subj)] = float(np.mean(accs))
    overall = float(np.mean(list(results.values()))) if results else float("nan")
    return {"per_subject": results, "mean": overall}


def leave_one_subject_out(build_pipeline_fn, X, y, groups, shuffle_labels=False, random_state=0):
    """LOSO CV. If shuffle_labels, permute y within each subject's group
    (independently per subject, preserving per-subject class balance) before
    fitting -- this is the permutation-baseline control.

    Returns per-subject accuracy dict, confusion matrices summed, and mean.
    """
    rng = np.random.RandomState(random_state)
    y_use = y.copy()
    if shuffle_labels:
        for subj in np.unique(groups):
            mask = groups == subj
            y_use[mask] = rng.permutation(y_use[mask])

    subjects = np.unique(groups)
    per_subject = {}
    cm_total = np.zeros((2, 2), dtype=int)
    for held_out in subjects:
        train_mask = groups != held_out
        test_mask = groups == held_out
        if len(np.unique(y_use[test_mask])) < 2:
            continue  # can't evaluate accuracy meaningfully without both classes
        pipe = clone(build_pipeline_fn())
        pipe.fit(X[train_mask], y_use[train_mask])
        preds = pipe.predict(X[test_mask])
        per_subject[int(held_out)] = float(accuracy_score(y_use[test_mask], preds))
        cm_total += confusion_matrix(y_use[test_mask], preds, labels=[0, 1])

    mean_acc = float(np.mean(list(per_subject.values()))) if per_subject else float("nan")
    return {"per_subject": per_subject, "mean": mean_acc, "confusion_matrix": cm_total.tolist()}


def group_kfold_cv(build_pipeline_fn, X, y, groups, n_splits=5):
    """GroupKFold: folds respect subject grouping but each fold contains
    multiple held-out subjects at once (faster than full LOSO, similar
    interpretation). Used as a cheaper stand-in when running many subjects.
    """
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    cm_total = np.zeros((2, 2), dtype=int)
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe = clone(build_pipeline_fn())
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
        cm_total += confusion_matrix(y[test_idx], preds, labels=[0, 1])
    return {"fold_accuracies": accs, "mean": float(np.mean(accs)), "confusion_matrix": cm_total.tolist()}


def subject_split_cv(
    build_pipeline_fn,
    X,
    y,
    groups,
    train_size=0.8,
    n_splits=10,
    random_state=0,
):
    """Subject-level train/test split at a configurable ratio, repeated
    n_splits times with different random subject partitions.

    Unlike within_subject_cv (splits trials within a subject) and unlike
    full leave_one_subject_out (always holds out exactly one subject),
    this lets you directly control what fraction of *subjects* go into
    training vs. testing -- e.g. train_size=0.5 for a 50/50 subject split,
    train_size=0.9 for a 90/10 split. Useful for studying how accuracy and
    its variance change as the number of training subjects shrinks or
    grows -- fewer training subjects should generalize worse and show
    higher fold-to-fold variance if the model is subject-sensitive.

    Returns fold-level accuracies (one per repeat) plus the mean/std, and
    how many subjects were in train vs. test for each fold.
    """
    gss = GroupShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=random_state)
    fold_results = []
    for train_idx, test_idx in gss.split(X, y, groups):
        pipe = clone(build_pipeline_fn())
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], preds)
        fold_results.append(
            {
                "accuracy": float(acc),
                "n_train_subjects": int(len(np.unique(groups[train_idx]))),
                "n_test_subjects": int(len(np.unique(groups[test_idx]))),
                "n_train_trials": int(len(train_idx)),
                "n_test_trials": int(len(test_idx)),
            }
        )
    accs = [f["accuracy"] for f in fold_results]
    return {
        "train_size": train_size,
        "folds": fold_results,
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
    }


def summarize(results: dict, title: str = "") -> str:
    lines = [f"=== {title} ===" if title else ""]
    if "per_subject" in results:
        accs = list(results["per_subject"].values())
        lines.append(f"n_subjects_evaluated = {len(accs)}")
        lines.append(f"mean accuracy        = {results['mean']:.3f}")
        if accs:
            lines.append(f"std                   = {np.std(accs):.3f}")
            lines.append(f"min / max             = {min(accs):.3f} / {max(accs):.3f}")
    if "confusion_matrix" in results:
        lines.append(f"confusion matrix (rows=true [left,right], cols=pred) =\n{np.array(results['confusion_matrix'])}")
    if "fold_accuracies" in results:
        lines.append(f"fold accuracies = {[round(a,3) for a in results['fold_accuracies']]}")
    return "\n".join(lines)
