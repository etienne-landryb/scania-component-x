# Component X — Cost-Sensitive Predictive Maintenance

Predicting imminent failure of an engine component in heavy-duty SCANIA trucks,
and turning that prediction into the **cheapest maintenance decision** rather
than the most likely one.

**[Live app](YOUR_STREAMLIT_URL)** · Data: [SCANIA Component X, IDA 2024 Industrial Challenge](https://arxiv.org/abs/2401.15199) (CC BY 4.0)

---

## Result

Evaluated on the challenge's held-out test set (5,045 vehicles) using its
official cost metric:

| Policy | Total cost |
|---|---:|
| Do nothing (assume every truck healthy) | 56,100 |
| Inspect every truck | 49,671 |
| **This model (cost-minimising)** | **33,994** |

**39.4% below doing nothing, 31.6% below inspecting everything.**
80% of imminent failures caught (48 of 60); 12 missed.

## The problem

Five-class, severely imbalanced, multivariate time-series classification.
Trucks emit irregular sensor readouts; from the history available "up to now"
the model assigns a class from 0 (no failure signal) to 4 (failure within 6
time units). 97% of vehicles are class 0.

Crucially, the scoring is **asymmetric**: calling an imminent failure healthy
costs 500, while an unnecessary workshop check costs 7–10. Accuracy is the
wrong objective — a model that predicts "healthy" every time scores 97%
accuracy and is worthless.

## Approach

**Labels.** The training file has no class column; each readout's class is
derived from its distance to the vehicle's repair event. Label-linked columns
are structurally excluded from the feature matrix to prevent leakage.

**Features.** Validation and test give one row per vehicle at a randomly chosen
"last readout", so training examples are built to match: sample a cut point,
summarise only readouts up to it, label with that readout's class. Four blocks
per example — current values, long-run accumulation rate, recent rate (which
catches acceleration near failure), and histogram *shape* normalised to
proportions so usage intensity doesn't dominate.

**Model.** LightGBM, early-stopped on **Total_cost** rather than log-loss.
This matters: under 97% class imbalance, log-loss is minimised almost
immediately by predicting the base rate, and early stopping fired at iteration
4 before this was fixed.

**Label-shift correction.** Uniform cut-point sampling produces a training
prior where class 1 outnumbers class 4 three to one; the challenge's
last-readout selection inverts that. Correcting the prior (a five-parameter
adjustment) improved calibration error from 0.0131 to 0.0089 and test cost from
36,516 to 33,994.

**Decision layer.** The model outputs probabilities; the action is whichever
minimises expected cost under the cost matrix:

```
E[cost | action m] = Σ_n  P(class n | x) · Cost[n][m]
```

This is why the imbalance is *not* handled by resampling — resampling distorts
the probabilities this rule depends on.

## Findings worth noting

**The economics collapse the label space.** The model almost never predicts
classes 1–3. Predicting "urgent" on a healthy truck costs 10; predicting
"schedule soon" on a truck about to fail costs 200. Hedging toward action
dominates, so a five-class problem becomes a binary operational decision.

**The decision layer carries the result, not the classifier.** Argmax scores
54,927 — barely below the naive baseline, because with a 97% majority class a
probability-maximising rule almost never predicts a minority class. Nearly all
the gain comes from the cost arithmetic.

## Limitations

- **Flags 44% of the fleet.** Optimal under the published cost matrix, but no
  workshop has that capacity. The app addresses this with an inspection-capacity
  constraint that ranks by cost avoided; the offline benchmark does not.
- **Class 3 recall is poor.** Most class-3 trucks are predicted class 4 (cheap,
  cost 7) or class 0 (expensive, cost 400). The 12 missed cost 4,800.
- **Calibration is improved, not solved.** Class 4 remains somewhat
  over-predicted after correction.
- **A cleaner fix exists.** Sampling training cut points to match the evaluation
  prior directly, rather than correcting afterwards, would be better. It means
  rebuilding features and is left as future work.
- **No deep or sequence models.** Published solutions used GNNs and cost-
  sensitive transformers. Matching them was not the goal; a well-engineered,
  honestly evaluated cost-aware baseline was.

## Repository

```
app.py              deployed Streamlit app (fleet triage + per-vehicle view)
demo/               small precomputed bundle the app reads
pipeline/           the offline pipeline, in run order
  run_phase1.py       ingestion + label construction
  run_phase2.py       cut-point sampling + feature engineering
  run_phase3.py       training, calibration, benchmark
  export_demo.py      builds demo/ from the trained model
```

The pipeline runs locally against the ~1.2 GB dataset; the app never touches it.
Probabilities are precomputed so the deployed app needs no ML dependencies —
the decision layer, which is this project's contribution, stays fully live.

## Running it

```bash
# app
pip install -r requirements.txt
streamlit run app.py

# pipeline (needs the dataset; set DATA_DIR in each runner)
pip install -r requirements-pipeline.txt
python pipeline/run_phase1.py && python pipeline/run_phase2.py && python pipeline/run_phase3.py
```
