# Methodology and interpretation

The business question is whether available application data can prioritize
vehicle-loan default risk under a limited review workload. This is an offline
benchmark, with no operational approval or intervention experiment.

## Data boundaries

1. Validate the labeled input, binary outcomes, and unique application IDs.
2. Split applications into 60% training, 20% validation, and 20% held-out test,
   stratified by outcome with random seed 42.
3. Explore labels only in training. Derive age, date, duration, balance, and ratio
   features per row; these operations use no other rows or labels.
4. Fit numerical imputation and smoothed categorical target encoding on training.
   `TargetEncoder.fit_transform` internally cross-fits the training representations
   across five stratified folds. Its `transform` uses training-only mappings for
   validation and test, and falls back to the training prior for unseen categories.
   This follows the documented
   [scikit-learn 1.4 TargetEncoder behavior](https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.TargetEncoder.html).
5. Fit the logistic-regression scaler on the encoded training matrix only. Train
   all five models on the same training rows, with explicit fixed hyperparameters.
6. Select the model with the highest validation ROC-AUC. Use alphabetical model
   name as a deterministic tiebreaker. Diagnostic challengers do not silently
   replace the primary model. No statistical significance claim is made.
7. Select an empirical score cutoff on validation that minimizes assumed error
   cost while flagging at most 25% of validation applications. Search thresholds
   from 0 to 1 in steps of 0.005, plus an endpoint above 1 to permit no flags.
   Equal-cost candidates use the larger threshold.
8. Freeze the selected model, preprocessing, and cutoff. Evaluate that model once
   on held-out test outcomes; also report the predeclared 0.50 reference cutoff.
   Do not refit on validation because that would change the score distribution
   after threshold selection.

The encoding folds are internal to preprocessing. Five-fold performance validation
of the selected family is now implemented on the training partition: each outer
fold refits the imputer, its internally cross-fitted target encoder, and the
estimator. No already-encoded full-training matrix is reused across outer folds.
The family was selected using separate validation data before these diagnostics;
this is not nested selection across all candidate model families. Fold sample
standard deviation describes fold variation rather than a future-performance
confidence interval.

## Exploration and controlled experiments

Fifteen training-only EDA charts cover data quality, class balance, financial
distributions, loan-size quantiles, LTV, bureau-score presence and descriptions,
employment, inquiries, recent delinquency, age, disbursal month, sourcing
concentration, Pearson correlations, and a joint LTV/bureau matrix. Count/rate
tables accompany the plots. Rate comparisons include pointwise 95% Wilson
intervals; no multiple-comparison or entity-correlation adjustment is claimed.
Histogram upper clipping at p99 and logarithmic count axes affect display only.

Controlled diagnostics compare the primary model against disabling class weights,
removing six sourcing/geographic IDs, and removing derived numeric features.
Reduced feature sets refit their own preprocessing. Two LightGBM parameter
challengers change leaf complexity or regularization. If another family is
selected, this LightGBM-specific challenger grid is skipped and recorded as such.
A fixed validation-only blend uses XGBoost/LightGBM/CatBoost weights of
0.35/0.35/0.30. Weights are not tuned.

Four seed comparisons change estimator randomness while holding the data split
and encoder fixed. They do not measure split uncertainty. Training AUC on the
in-sample cross-fitted representation is an optimistic fitting diagnostic.
All these experiments use training and validation only. The primary five-model
selection remains the declared benchmark; diagnostic improvements do not trigger
a test-based replacement.

Forty-five policy scenarios combine review caps of 10/20/25/35/50%, loss
assumptions of 25/45/65%, and lost margins of 4/8/12%. They reuse validation
confusion counts and preserve a no-review endpoint for tied scores.

Test ranking and policy intervals use 200 stratified row-bootstrap samples of
the frozen model's predictions. The pointwise percentile intervals condition on
observed class prevalence and omit fitting uncertainty and entity dependence.

## Decision scenario

The validation policy assumes:

- Missed-default cost: 45% of the average loan amount in training.
- Flagged non-default cost: 8% of that same average loan amount.
- Review capacity: at most 25% of validation applications flagged.

Scenario cost is `FN * cost_FN + FP * cost_FP`. Approving all applications has
scenario cost `actual_defaults * cost_FN`. The relative cost change is the
difference divided by the approve-all cost. Currency is not reinterpreted or
converted; values are in the dataset's nominal amount units.

This simplified scenario treats every flagged default as a fully prevented loss
and charges margin for every flagged non-default. Real review neither guarantees
loss prevention nor necessarily rejects a good applicant. There is no measured
review cost, recovery rate, revenue model, or intervention response. The scenario
is illustrative and cannot support a realized-savings claim.

The cap applies to validation data. A fixed cutoff may exceed 25% workload on test
or future data; that outcome should be reported, not corrected using test labels.
This model does not implement a fixed-volume top-k queue.

## Offline deployment and monitoring demonstrations

Three queues use a frozen high cutoff and a validation-fitted low cutoff. The
low cutoff maximizes the eligible threshold below the high cutoff subject to
at least 5% of validation applications, at least 30 cases, and at most 10%
observed validation default rate in the lower queue. If no candidate qualifies,
that queue is disabled. The fitted rate is not a guarantee on new data. Every
queue retains ordinary human underwriting; no automatic lending action follows.

Monitoring code computes numerical PSI using training quantile boundaries,
infinite end bins, an explicit missing bin, and smoothed population shares.
It also reports unseen categorical shares and missingness. Training-to-random-
validation comparisons demonstrate the calculations; they are not prospective
drift evidence. There is no universal alert threshold, live scheduler, service,
registry, or feedback integration. The CSV monitoring plan describes future work.

## Limits and next evaluation

- Scores from class-weighted classifiers are uncalibrated. They support ranking
  and empirical routing thresholds, not direct probability-of-default claims.
- Model and cutoff selection share a validation sample. Nested validation or an
  additional policy-development partition could quantify selection uncertainty.
- The source dataset was previously explored. This random holdout is an internal
  benchmark rather than an untouched external dataset.
- A random application split can share dealers, staff, and regions across sets.
  Sourcing IDs can be strong proxies for historical patterns. Use future-period
  and unseen-entity holdouts to evaluate transfer to changed populations.
- Age, location, document availability, and sourcing fields require subgroup,
  fairness, privacy, and decision-time availability review before lending use.
- Native importance is descriptive and estimator-specific. It is not a causal
  effect, an individual explanation, or a comparison across model families.
- Calibration, longitudinal drift, fitting uncertainty, intervention effects,
  and operational costs remain future work. Implemented bootstrap intervals and
  random-split monitoring checks have the narrower scopes described above.
