# Local validation

Verified on 12 November 2026 using Python 3.11.14 and the versions recorded in
`metrics.json` and `requirements.txt`.

- Full-data notebook: all 19 code cells across 10 stages executed successfully,
  with no error outputs.
- Dataset: 233,154 labeled applications; 139,892 training, 46,631 validation,
  and 46,631 held-out test applications.
- Synthetic unit checks: 14 passed, covering disjoint reproducible splits,
  cross-fitted encoding, unseen categories and train-only medians, row-local
  feature engineering, validation capacity, threshold cost minimization,
  tied-score feasibility, both error costs, invalid configuration, feature-set
  ablations, Wilson intervals, PSI changes/missingness, routing boundaries,
  low-queue feasibility, 45-scenario capacity constraints, and bootstrap repeatability.
- Extended full-data diagnostics: five outer folds with preprocessing refitted
  per fold; four fixed-split estimator seeds; three controlled ablations; two
  LightGBM parameter challengers; a fixed weighted ensemble; and 45 policy scenarios.
- Training-only exploration: 15 chart artifacts plus aggregate quality, counts,
  rate intervals, financial distributions, concentration, and correlation tables.
- Frozen test uncertainty: 200 stratified bootstrap samples, conditional on the
  fitted model and observed prevalence.
- Offline operations: three-queue routing with validation-fitted cutoffs;
  reference-bin PSI, missingness, and unseen-category monitoring demonstrations.
- CLI: completed a separate 5,000-application sample run; outputs were written
  to ignored `reports/smoke/`, preserving the full-data reports.
- Artifact checks: notebook schema valid, no unexecuted code cells, source
  fingerprint consistent, confusion counts and costs consistent, README result
  consistent, and local Markdown links resolve.
- Publication review: Git ignore rules exclude raw CSV/XLSX inputs, sample
  outputs, virtual environments, and environment files. The publication folder
  contains aggregate outputs and no individual applicant previews or scores.

Published reproduction commands from the analysis folder:

```powershell
python -m unittest discover -s tests -v
python src/credit_risk.py --data data/raw/train.csv --output reports
python src/credit_risk.py --data data/raw/train.csv --sample 5000 --output reports/smoke
```

Notebook rebuild and artifact checks were also run using local helpers in
`scripts/`, which are excluded by current Git ignore rules. The saved notebook
can be opened and re-executed in a Jupyter-compatible editor.

The CLI sample run checks the execution path; its metrics do not validate the
full-data model. The full-data notebook provides the published numerical evidence.
A fresh virtual-environment installation, future-period performance, external
validation, fairness assessment, longitudinal drift, and production integration
have not been verified.
