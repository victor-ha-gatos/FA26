"""
CSP + LDA pipeline definition.

CSP (Common Spatial Patterns) finds spatial filters that maximize the
variance ratio between two classes -- exactly the tool the original
Schalk et al. / MNE tutorials use for this dataset, and a natural fit here:
motor imagery/execution modulates band power (mu/beta ERD) over
contralateral sensorimotor cortex, which is a *variance* effect, not a
mean-amplitude effect. LDA on log-variance CSP components is the standard
follow-up classifier: few features, linear, hard to overfit with ~50-100
trials per subject.
"""
from __future__ import annotations

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


def build_pipeline(n_components: int = 4, reg: str | None = "ledoit_wolf") -> Pipeline:
    """CSP -> LDA pipeline.

    n_components=4: 2 pairs of filters (largest/smallest eigenvalue ends of
    the CSP spectrum), the standard choice in the MNE tutorial this task
    references. reg='ledoit_wolf' shrinkage-regularizes the covariance
    estimates CSP relies on; raw covariance is unstable with only ~1-2s
    windows and 64 channels (64x64 covariance from ~160 time samples is
    close to rank-deficient before filtering/cropping).
    """
    csp = CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    return Pipeline([("csp", csp), ("lda", lda)])
