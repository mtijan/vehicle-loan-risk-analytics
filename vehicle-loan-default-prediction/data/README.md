# Data

The experiment uses the labeled vehicle-loan application dataset associated with
the [LTFS Data Science FinHack hosted by Analytics Vidhya](https://www.analyticsvidhya.com/datahack/contest/ltfs-datascience-finhack-an-online-hackathon/).
The local source files were obtained from the
[Vehicle Loan Default Prediction mirror on Kaggle](https://www.kaggle.com/datasets/avikpaul4u/vehicle-loan-default-prediction).
Provider attribution does not imply affiliation or endorsement.

Download the data from the provider or mirror, review its applicable access and
usage terms, and place `train.csv` in `data/raw/`. The CSV itself is not distributed
in this repository. No license for third-party data is asserted by this project.

```text
data/
  README.md
  raw/
    .gitkeep
    train.csv               # required, local and ignored
    test.csv                # optional, unlabeled and unused
    Data Dictionary.xlsx    # optional provider reference, local and ignored
```

The supplied labeled CSV has 233,154 applications, 41 columns including the
application ID and target, and 50,611 default outcomes. Column names in this local
version are uppercase with underscores; the script expects this schema. The
input SHA-256 is recorded in `reports/metrics.json` so a different download can
be identified.

`LOAN_DEFAULT` is the provider-supplied binary outcome. It should not be interpreted
as lifetime default, a measured loss amount, or proof that an application should
be rejected. `UNIQUEID` is used to validate unique applications and is excluded
from modeling. The original `test.csv` has no outcome labels; reported test
metrics come from a held-out partition of the labeled `train.csv`.

Inputs cover loan and asset amounts, LTV, demographic and disbursal dates,
employment, bureau scores, account balances and arrears, account-history
durations, sourcing identifiers, geography, and document indicators. Raw dates,
application IDs, and raw account-duration strings are removed after feature
engineering. See the provider dictionary for detailed field meanings.

Only aggregate summaries and model-level results are published. Notebook outputs
do not include individual applicant records, row-level scores, or data previews.
