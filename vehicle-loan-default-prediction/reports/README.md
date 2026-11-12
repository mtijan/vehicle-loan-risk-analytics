# Results and evidence

The committed results come from executing every code cell in
`notebooks/vehicle_loan_default_prediction.ipynb` on the full local labeled dataset.
The notebook calls the same `run_experiment` function as the Python CLI.

The main [data science report](../README.md) interprets those results across the
ten-stage lifecycle: why each EDA question was chosen, what the evidence shows,
how model experiments compare, and which business actions need prospective
measurement. The report is based on saved JSON/CSV artifacts. Generator and
verification helpers under `scripts/` are local and excluded by current Git
ignore rules. Published reproduction uses `src/credit_risk.py` or the notebook.

| Artifact | Meaning |
| --- | --- |
| `metrics.json` | Input fingerprint, versions, configuration, splits, chosen model, validation policy, and test results |
| `tables/model_comparison.csv` | Five models compared on validation ROC-AUC and average precision |
| `tables/validation_thresholds.csv` | Validation-only threshold grid, workload, recall, precision, and scenario cost |
| `tables/test_segments.csv` | Aggregate outcomes above and below the frozen cutoff |
| `tables/ltv_default_rates.csv` | Training-only default rates and sample counts by LTV band |
| `tables/feature_importance.csv` | Native importance or absolute standardized coefficients, labeled by type |
| `eda_summary.json` | Fifteen-chart EDA inventory, training window, data-quality notes |
| `tables/data_quality.csv`, `financial_summary.csv` | Schema-level quality and aggregate financial distribution summaries |
| Other EDA tables | Training-only rates, pointwise 95% Wilson intervals, counts, correlations, and concentration |
| `experiments.json` | Extended diagnostics summary and boundaries |
| `tables/ablation_experiments.csv` | Weighting, sourcing/geography, and derived-feature ablations |
| `tables/hyperparameter_challengers.csv` | Small validation-only LightGBM challenger grid when that family is selected |
| `tables/ensemble_validation.csv` | Fixed weighted blend vs primary selected model on validation |
| `tables/cross_validation.csv` | Five outer folds on training with fresh preprocessing per fold |
| `tables/seed_stability.csv`, `generalization_gap.csv` | Estimator-seed stability and descriptive fitting gap |
| `tables/cost_capacity_sensitivity.csv` | Forty-five validation-only assumption combinations |
| `tables/test_bootstrap_intervals.csv` | Pointwise stratified-bootstrap intervals, conditional on prevalence and frozen model |
| `routing_policy.json` | Frozen queue thresholds, actions, and monitoring-demo scope |
| `tables/validation_routing.csv`, `test_routing.csv` | Three queues with aggregate outcomes and Wilson intervals |
| `tables/monitoring_feature_baseline.csv`, `monitoring_unseen_categories.csv` | Training-reference PSI, missingness, and unseen-category validation baseline |
| `tables/monitoring_plan.csv` | Proposed future monitoring process; no live service |
| `tables/business_recommendations.csv` | Notebook-generated recommendations connected to computed evidence |
| `figures/*.png` | Conventional labeled charts produced by the Python pipeline |

Re-running the experiment replaces these aggregate artifacts. The configuration
and input fingerprint determine which experiment is represented; outputs are not
a production monitoring record. Training times can differ between machines.

The held-out test set is an internal random-split benchmark. It is not the
unlabeled competition test file, an out-of-time evaluation, or external validation.
Cost changes are hypothetical and depend on assumed loss and margin rates.

For a shorter development run, keep its outputs separate:

```powershell
python src/credit_risk.py --data data/raw/train.csv --sample 5000 --output reports/smoke
```

Sample outputs are ignored by Git and are not evidence for the full-data results.
Use `--quick` to skip extended model diagnostics while retaining five-model
training, EDA, routing, and monitoring. A quick run cannot substitute for the
complete committed experiment inventory.
