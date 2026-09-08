# FA26 — Left vs. Right Fist EEG Classification: Notes, Status, ToDo

## What this is
Working notes for the FA26 recruitment challenge (motor imagery/execution,
left vs. right fist, PhysioNet EEGMMIDB). This file is the running log +
open-questions list the challenge asks for ("log your attempts").

## Data, as found
- `data/` = EEGMMIDB, already downloaded: 109 subjects (`S001`-`S109`),
  each with 14 runs (`S###R01.edf`...`S###R14.edf`) + matching `.edf.event`
  files (ground-truth annotations, redundant with the annotations embedded
  in the EDF itself — used both as a cross-check).
- Confirmed via `mne.io.read_raw_edf`: 64 EEG channels, nominally 160 Hz,
  annotations `T0`/`T1`/`T2` per run.
- Run mapping (confirmed against the PDF instructions + by checking which
  runs actually contain T1/T2 vs. only T0):
  - Runs 3, 7, 11: **executed** left/right fist. T1=left, T2=right.
  - Runs 4, 8, 12: **imagined** left/right fist. T1=left, T2=right.
  - Runs 5,9,13 / 6,10,14: both-fists-vs-both-feet (not used for this task).
  - Runs 1, 2: eyes-open/closed baseline (no T1/T2, not used).
- **This task only uses T1/T2 trials, never T0 (rest)** — the assignment
  is left-fist-vs-right-fist, not fist-vs-rest.

### Known data-quality issues (found by inspection, not assumed)
- Subjects **88, 92, 100** are recorded at **128 Hz**, not 160 Hz, and/or
  have a different trial duration (checked S088/S092/S100 vs. others).
  Handled by resampling every file to 160 Hz in `data.py::load_raw`.
- Trial counts per run are not always exactly 15 (e.g., S034/S037 executed
  runs have 14 trials instead of 15) — handled naturally since the loader
  counts actual annotations rather than assuming a fixed count.
- Did **not** exhaustively QC all 109 subjects' impedance/noise levels —
  see ToDo below. There are documented issues with a handful of EEGMMIDB
  subjects (channel corruption, mislabeled events) reported by other
  groups using this dataset; I have not cross-checked against those
  reports.

## Data - Deep Dive
### Folder Structure
data/ \
|-- ANNOTATORS                    \
|-- RECORDS                       \
|-- 64_channel_sharbrough.pdf     \
|-- S001/                         \
|-- |-- S001R01.edf               \
|-- |-- S001R01.edf.event         \
|-- S002/  ... (same pattern)     \
|-- ...                           \
|--  S109/  ... (same pattern)

### Files
For this project, only Runs # 3, 4, 7, 8, 11, and 12 are used.

| Run | Task | Condition | Subjects (.edf files) | .edf.event files | 
|-----|------|-----------|-----------------------|------------------|
|03 | Left vs. right fist | Executed | 109 | 109 |
|04 | Left vs. right fist | Imagined | 109 | 109 |
|07 | Left vs. right fist | Executed | 109 | 109 |
|08 | Left vs. right fist | Imagined | 109 | 109 |
|11 | Left vs. right fist | Executed | 109 | 109 |
|12 | Left vs. right fist | Imagined | 109 | 109 |

.edf (European Data Format): Actual Signal
  - Standard binary file format for storing multi-channel biosignal recordings (EEG, EMG, ECG, etc.).
  - Contains: a header (subject metadata, recording start time, number of channels, channel labels, sample rate, physical/digital signal ranges) followed by  the raw digitized signal samples (16bit signed integer) for every channel.
  - In this dataset: 64 EEG channels sampled at 160 Hz, continuous recording for the full run duration (~2 minutes for runs 3-14).
  - Annotations (event markers T1/T2) are also embedded inside the EDF itself as an "EDF+ Annotations" channel

.edf.event: Separate Annotations (e.g., T1: Left Fist, T2: Right Fist)

### Data Visualization
- Digital Value: 
  - Each raw digitaized signal sample (16bit signed intger) stored in edf file is a measurement value from the EEG electrode
    - An electrode picks up electrical potential from neural activity
    - Amplifier boosts the signal
    - Anti-aliasing Filter is applied to prepare for conversion to digital values
    - ADC converts analog values to digital values sampled at 160Hz
  - Data has been collected from 64 electrodes (channels) as shown in ***64_channel_sharbrough.pdf***
  - Each .edf file contains 64 parallel samples captured at 160Hz
- Physical Value: 
  - Converted from Digital Value to microvolts ($\mu V$) of electrical potential at the scalp
    - Conversion Equation: 
  - Visualization of a single channel/electrode (C3) from one subject (S001) during Run #4
    - x axis: time in second
    - y axis: physical value in microvolts $\mu V$
    - colored band: Annotation value (T0, T1, T2)

    <img src="results/plots/single_channel_raw.png" alt="Single-channel raw EEG signal" width="30%">

## Environment
- Set up a project-local venv at `~/FA26/.venv` 
- `requirements.txt` has pinned versions
  - This file lists all the packages that need to be installed for the project to run
- To reproduce, run on Terminal:
  - cd ~/FA26
  - python3.11 -m venv .venv
  - .venv/bin/pip install -r requirements.txt
  - source .venv/bin/activate

## Pipeline (`src/`)
- `data.py` 
  - loads EDF files
  - standardizes channel names/montage
  - resamples to 160Hz (For subjects 88, 92, and 100 that were sampled at 120Hz)
  - band-pass filters 7-30 Hz (mu+beta rhythms - where sensorimotor ERD/ERS during fist movement/imagery shows up)
  - epochs on T1/T2 only (tmin=-1s, tmax=4s around cue)
  - crops to a 1-2s post-cue window for CSP fitting (standard MNE-tutorial window, avoids cue-onset transient and movement/imagery offset).
- `model.py` — `CSP(n_components=4, reg='ledoit_wolf', log=True) -> LDA`.
  CSP finds spatial filters maximizing between-class variance ratio, which
  is the right primitive here because the physiological signal (ERD/ERS)
  is a *band-power* (variance) effect, not a mean-amplitude effect.
  Ledoit-Wolf shrinkage regularizes the 64x64 spatial covariance, which is
  numerically unstable to estimate from ~160 time samples per trial without it.
- `evaluate.py` — three evaluation regimes (see "Evaluation design" below).
- `train.py` — orchestrates load -> evaluate -> save model + json results.
- `predict.py` — deliverable #2: `predict_edf(model_path, edf_path)` /
  CLI. Loads a saved model, epochs a raw EDF the same way as training, and
  returns per-trial predicted left/right fist labels; reports accuracy if
  the file's own T1/T2 labels are available for a live check.
- `plot_results.py` — per-subject accuracy scatter (real vs. permuted
  labels), saved to `results/loso_accuracy.png`.

## Evaluation design (why three regimes, not one number)
1. **Within-subject CV** (`ShuffleSplit`, 80/20, 5 repeats, per subject):
   most optimistic number. CSP/LDA get to fit that person's exact head
   geometry/noise. Useful as an upper bound, not as evidence of
   generalization.
2. **Leave-one-subject-out (LOSO)**: fit on N-1 subjects, test on the held
   out one, repeated for all subjects. This is the number that matters for
   "does this work on a new person" — the framing the prompt cares about.
3. **Permutation baseline**: same LOSO procedure, but labels shuffled
   *within each subject* before fitting (preserves per-subject class
   balance, breaks the left/right correspondence). If real ≈ permuted,
   the real number isn't measuring left-vs-right content.

All subject-grouped splits use subject IDs as groups so no subject's
trials appear in both train and test in the same fold — the single most
important control here, since within-subject leakage (same head, same
session noise) is confounded with the fist label unless explicitly split
out.

## Results so far (n=40 subjects, S001-S040)

| Regime | Executed (3,7,11) | Imagined (4,8,12) |
|---|---|---|
| Within-subject CV | 0.579 (± 0.122) | 0.536 (± 0.164) |
| **LOSO, real labels** | **0.543 (± 0.070)** | **0.546 (± 0.092)** |
| LOSO, permuted labels | 0.478 (± 0.048) | 0.503 (± 0.049) |
| Paired t-test (real vs. permuted, per-subject) | t=4.26, p=0.0001 | t=2.85, p=0.007 |

See `results/loso_accuracy.png` for the full per-subject spread (sorted),
`results/results_executed_40.json` / `results_imagined_40.json` for raw
numbers, `results/model_executed_40.joblib` / `model_imagined_40.joblib`
for the saved pipelines predict.py loads.

**Reading of these numbers:**
- LOSO real accuracy (~0.54-0.55) is statistically distinguishable from
  the permutation baseline (~0.48-0.50) — there is a real, if weak,
  cross-subject left/right signal being picked up, consistent with the
  literature's characterization of this task as "weak, noisy, unreliable."
- Executed and imagined are statistically similar at this scale (both
  ~0.54), which was mildly surprising — I expected executed to show a
  clearer LOSO advantage given its stronger single-subject signal
  (within-subject CV is a bit higher for executed: 0.579 vs 0.536). More
  subjects and/or a paired same-subject executed-vs-imagined comparison
  would sharpen this.
- Per-subject spread is wide (min 0.467/0.378, max 0.778/0.822) — a
  single mean accuracy badly understates how variable this is. Some
  subjects are near chance under real labels; a few are much better than
  the mean. This is exactly the "range, not best case" the prompt asks for.
- Within-subject CV overstates generalizable performance relative to
  LOSO, as expected, but the gap is smaller than I initially guessed —
  suggesting the CSP+LDA model isn't wildly overfitting to individual
  subject idiosyncrasies once shrinkage-regularized; it's just that the
  cross-subject signal itself is weak.

## What I have NOT yet done (ToDo)
- [ ] Scale LOSO to the full 109 subjects (currently n=40, chosen for
      turnaround time in this pass — training + evaluating all 109 with
      full LOSO is ~3x this runtime; queued as a follow-up run).
- [ ] Quantify "how much of this is about the person, not the task" more
      directly: e.g., train a subject-identity classifier on the same
      features and see how well subject ID alone is decodable, to bound
      how much of any left/right signal could ride on subject-correlated
      artifacts. Currently I only have the permutation-baseline control,
      which addresses label-shuffling but not a full subject-confound
      audit (e.g., electrode impedance differences, session-to-session
      amplifier drift).
- [ ] Test whether a model trained on **executed** runs transfers to
      **imagined** runs (and vice versa) — explicitly suggested in the
      prompt as "fair game and possibly interesting." Not yet run.
- [ ] Try a channel subset restricted to sensorimotor electrodes (C3, C4,
      Cz and neighbors) instead of all 64 — both as a sanity check (does
      CSP concentrate its spatial patterns there, as physiology predicts?)
      and as a way to test robustness to electrode-count mismatches on
      other EDF files.
- [ ] Look at what CSP is actually learning: plot the spatial patterns
      (`CSP.plot_patterns`) for a few subjects to check whether the
      topography is plausible (contralateral sensorimotor, roughly
      symmetric between left/right) rather than picking up an artifact
      (e.g., a bad channel, eye-movement contamination near frontal
      electrodes).
- [ ] Sweep the crop window (currently fixed 1-2s post-cue) and the
      band-pass range (currently fixed 7-30 Hz) — both were taken from the
      standard MNE tutorial defaults referenced in the challenge PDF, not
      independently tuned or justified for this dataset.
- [ ] Cross-check accuracy against a hand-crafted spectral-power baseline
      (log-bandpower in mu/beta per channel -> LDA, no CSP) to isolate how
      much CSP's spatial filtering specifically contributes versus a
      simpler feature.
- [ ] Full data QC pass: I have not screened all 109 subjects for bad
      channels / flat channels / motion artifact beyond the coarse
      sfreq/trial-count check already done for S088/S092/S100. Some
      literature on EEGMMIDB reports specific subjects with known
      recording issues — I have not cross-referenced that list.
- [ ] `predict.py` currently assumes the EDF has all 64 standard channels
      in a nameable/standardizable form (via `mne.datasets.eegbci.standardize`).
      Have not tested it against a file with a different channel montage
      or a partial channel set — worth hardening before calling it a real
      deliverable-grade CLI.
- [ ] Demo video, README with pinned one-line usage example, and GitHub
      repo publication are not done yet — this session focused on the
      technical pipeline and its evaluation.

## Open questions / things I'd want to defend or am least sure of
- Is 1-2s post-cue the right window for *executed* trials specifically?
  Movement onset latency for executed fist clenches may differ from
  imagined ones; using the same fixed crop for both may quietly
  disadvantage one condition. Not yet checked.
- The permutation baseline shuffles within-subject, which controls for
  subject-level confounds but not for run-order or fatigue/drift-within-
  run confounds (e.g., if T1 trials cluster earlier in a run than T2 for
  incidental protocol reasons, permuting within-subject-across-all-trials
  won't catch a within-run temporal bias). Would want to also try
  shuffling within-run rather than within-subject to rule this out.
- LOSO with only 40 subjects gives per-subject accuracy estimates with
  meaningful sampling noise (as few as ~42-45 trials per held-out
  subject for executed runs). The between-subject std (~0.07-0.09) partly
  reflects subject differences and partly reflects small-n measurement
  noise per subject — haven't decomposed those two sources.
- Haven't yet checked whether accuracy correlates with anything
  measurable about a subject (session length, recording date, channel
  noise level) that would indicate a data-quality confound rather than a
  genuine physiological difference in imagery/execution ability.

## AI tool use
Used this AI assistant (Enchanté, running Claude) to read the challenge
PDFs, inspect the dataset structure interactively (EDF header content, run
annotation labels, sfreq irregularities), scaffold the pipeline code above,
and run/summarize experiments. One specific choice made and reasoned about
directly (not just accepted from the tutorial default): using
Ledoit-Wolf-shrinkage-regularized CSP covariance (`reg='ledoit_wolf'`)
rather than the tutorial's unregularized default (`reg=None`) — the
64-channel spatial covariance is estimated from only ~160 time samples per
trial after cropping to the 1s analysis window, which is close to
rank-deficient (fewer samples than channels means the raw sample
covariance is not even guaranteed full rank), so shrinkage was chosen
deliberately rather than left at the tutorial's default.
