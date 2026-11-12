"""Reproducible vehicle-loan risk experiment with separate validation and test sets.

Run from the repository root: python src/credit_risk.py --data data/raw/train.csv
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder


TARGET = "LOAN_DEFAULT"
CATEGORICAL = [
    "SUPPLIER_ID", "BRANCH_ID", "STATE_ID", "MANUFACTURER_ID",
    "CURRENT_PINCODE_ID", "EMPLOYEE_CODE_ID", "PERFORM_CNS_SCORE_DESCRIPTION",
    "EMPLOYMENT_TYPE",
]
DROP_COLUMNS = [
    "UNIQUEID", TARGET, "DATE_OF_BIRTH", "DISBURSAL_DATE",
    "AVERAGE_ACCT_AGE", "CREDIT_HISTORY_LENGTH",
]


@dataclass(frozen=True)
class Config:
    seed: int = 42
    test_size: float = 0.20
    validation_size: float = 0.20
    max_review_share: float = 0.25
    loss_given_default: float = 0.45
    lost_margin: float = 0.08
    jobs: int = 4
    sample: int | None = None
    extended: bool = True

    def __post_init__(self):
        if not (0 < self.test_size < 1 and 0 < self.validation_size < 1
                and self.test_size + self.validation_size < 1):
            raise ValueError("Split fractions must be positive and sum to less than one.")
        if not 0 < self.max_review_share <= 1:
            raise ValueError("max_review_share must be in (0, 1].")
        if not (0 < self.loss_given_default <= 1 and 0 <= self.lost_margin <= 1):
            raise ValueError("Cost assumptions must be fractions between zero and one.")


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create row-level features without learning from other rows or their labels."""
    x = raw.drop(columns=DROP_COLUMNS, errors="ignore").copy()
    for source, name in [
        ("AVERAGE_ACCT_AGE", "AVG_ACCT_AGE_MONTHS"),
        ("CREDIT_HISTORY_LENGTH", "CREDIT_HISTORY_MONTHS"),
    ]:
        duration = raw[source].astype("string").str.extract(r"^(\d+)yrs\s*(\d+)mon$")
        x[name] = pd.to_numeric(duration[0]) * 12 + pd.to_numeric(duration[1])
    birth = pd.to_datetime(raw["DATE_OF_BIRTH"], format="%d-%m-%Y", errors="coerce")
    disbursed = pd.to_datetime(raw["DISBURSAL_DATE"], format="%d-%m-%Y", errors="coerce")
    age = (disbursed - birth).dt.days / 365.25
    x["AGE_AT_DISBURSAL"] = age.where(age.between(18, 100))
    x["DISBURSAL_MONTH"] = disbursed.dt.month
    x["DISBURSAL_DAY"] = disbursed.dt.day
    x["DISBURSAL_WEEKDAY"] = disbursed.dt.dayofweek
    x["DOWN_PAYMENT"] = raw["ASSET_COST"] - raw["DISBURSED_AMOUNT"]
    x["DISBURSED_TO_ASSET_RATIO"] = raw["DISBURSED_AMOUNT"] / raw["ASSET_COST"].replace(0, np.nan)
    x["TOTAL_ACTIVE_ACCTS"] = raw["PRI_ACTIVE_ACCTS"] + raw["SEC_ACTIVE_ACCTS"]
    x["TOTAL_OVERDUE_ACCTS"] = raw["PRI_OVERDUE_ACCTS"] + raw["SEC_OVERDUE_ACCTS"]
    x["TOTAL_CURRENT_BALANCE"] = raw["PRI_CURRENT_BALANCE"] + raw["SEC_CURRENT_BALANCE"]
    x["OVERDUE_RATIO"] = x["TOTAL_OVERDUE_ACCTS"] / (raw["PRI_NO_OF_ACCTS"] + raw["SEC_NO_OF_ACCTS"] + 1)
    x["NO_BUREAU_SCORE"] = raw["PERFORM_CNS_SCORE"].eq(0).astype(int)
    x["INSTALLMENT_TO_LOAN"] = raw["PRIMARY_INSTAL_AMT"] / raw["DISBURSED_AMOUNT"].replace(0, np.nan)
    x["PRI_UTILIZATION"] = raw["PRI_CURRENT_BALANCE"] / raw["PRI_SANCTIONED_AMOUNT"].replace(0, np.nan)
    for col in CATEGORICAL:
        x[col] = x[col].fillna("Unknown").astype(str)
    return x.replace([np.inf, -np.inf], np.nan)


def load_data(path: Path, config: Config) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if TARGET not in raw:
        raise ValueError("Use the labeled train.csv; competition test.csv has no outcomes.")
    if raw[TARGET].isna().any() or set(raw[TARGET].unique()) != {0, 1}:
        raise ValueError("LOAN_DEFAULT must contain both binary classes with no missing labels.")
    if raw["UNIQUEID"].isna().any() or raw["UNIQUEID"].duplicated().any():
        raise ValueError("Application IDs must be present and unique before splitting.")
    if config.sample is not None:
        if not 50 <= config.sample <= len(raw):
            raise ValueError("sample must be between 50 and the dataset row count.")
        if config.sample < len(raw):
            raw, _ = train_test_split(raw, train_size=config.sample,
                                      stratify=raw[TARGET], random_state=config.seed)
    return raw.reset_index(drop=True)


def split_data(raw: pd.DataFrame, config: Config):
    """Split before fitting encoders, imputers, models, or choosing a threshold."""
    development, test = train_test_split(
        raw, test_size=config.test_size, stratify=raw[TARGET], random_state=config.seed
    )
    train, validation = train_test_split(
        development,
        test_size=config.validation_size / (1 - config.test_size),
        stratify=development[TARGET], random_state=config.seed,
    )
    return train.copy(), validation.copy(), test.copy()


def preprocess(x: pd.DataFrame, seed: int) -> ColumnTransformer:
    categorical = [c for c in CATEGORICAL if c in x]
    numeric = [c for c in x if c not in CATEGORICAL]
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("category", TargetEncoder(target_type="binary", smooth=20.0,
                                   cv=5, shuffle=True, random_state=seed), categorical),
    ], verbose_feature_names_out=False)


def build_models(y: pd.Series, config: Config):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    weight = float((y == 0).sum() / (y == 1).sum())
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=2000,
                                         class_weight="balanced", random_state=config.seed)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", n_jobs=config.jobs, random_state=config.seed),
        "XGBoost": XGBClassifier(
            n_estimators=1000, learning_rate=0.015, max_depth=5, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=5, gamma=0.5,
            scale_pos_weight=weight, eval_metric="auc", tree_method="hist",
            n_jobs=config.jobs, random_state=config.seed),
        "LightGBM": LGBMClassifier(
            n_estimators=1200, learning_rate=0.01, num_leaves=31,
            min_child_samples=100, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, reg_lambda=5, reg_alpha=1,
            class_weight="balanced", n_jobs=config.jobs, random_state=config.seed, verbosity=-1),
        "CatBoost": CatBoostClassifier(
            iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=10,
            auto_class_weights="Balanced", random_seed=config.seed,
            thread_count=config.jobs, verbose=False, allow_writing_files=False),
    }


def evaluate(y, scores, threshold: float, cost_fn: float, cost_fp: float) -> dict:
    predicted = np.asarray(scores) >= threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    cost = fn * cost_fn + fp * cost_fp
    baseline = (tp + fn) * cost_fn
    return {
        "threshold": float(threshold), "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else 0.0,
        "specificity": tn / (tn + fp), "accuracy": (tn + tp) / len(y),
        "review_share": float(predicted.mean()), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "scenario_cost": cost, "approve_all_cost": baseline,
        "scenario_cost_reduction": (baseline - cost) / baseline if baseline else 0.0,
    }


def choose_threshold(y, scores, config: Config, cost_fn: float, cost_fp: float):
    """Minimize assumed validation cost subject to a validation review-capacity cap.

    Include the all-negative endpoint to ensure a feasible choice even with tied scores.
    No test labels or scores are accepted by the training workflow here.
    """
    candidates = np.r_[np.linspace(0, 1, 201), np.nextafter(1.0, 2.0)]
    # Ranking metrics are constant across thresholds; calculate them once.
    auc = float(roc_auc_score(y, scores))
    ap = float(average_precision_score(y, scores))
    rows = []
    y = np.asarray(y)
    scores = np.asarray(scores)
    for threshold in candidates:
        pred = scores >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        rows.append({"threshold": float(threshold), "review_share": float(pred.mean()),
                     "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                     "recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else 0.0,
                     "specificity": tn / (tn + fp), "scenario_cost": fn * cost_fn + fp * cost_fp,
                     "roc_auc": auc, "average_precision": ap})
    table = pd.DataFrame(rows)
    eligible = table[table.review_share <= config.max_review_share]
    chosen = eligible.sort_values(["scenario_cost", "threshold"], ascending=[True, False]).iloc[0]
    return float(chosen.threshold), table


def plot_results(comparison, y_test, scores, policy, threshold_table, output, config):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ranked = comparison.sort_values("validation_roc_auc")
    bars = ax.barh(ranked.model, ranked.validation_roc_auc, color="#32658a")
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.axvline(0.5, color="#555555", linestyle="--", label="Random ranking: 0.50")
    ax.set(xlim=(0, 1), xlabel="Validation ROC-AUC (0 to 1)", title="Model comparison on validation data")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "figures/model_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y_test, scores)
    axes[0].plot(fpr, tpr, color="#32658a", label=f"Selected model: AUC {policy['roc_auc']:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#777777", label="Random ranking")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out test ROC", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend()
    precision, recall, _ = precision_recall_curve(y_test, scores)
    axes[1].plot(recall, precision, color="#32658a", label=f"Average precision {policy['average_precision']:.4f}")
    axes[1].axhline(np.mean(y_test), linestyle="--", color="#777777", label="Test prevalence")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Held-out test precision-recall", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "figures/test_ranking.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(threshold_table.review_share * 100, threshold_table.recall * 100, color="#32658a")
    ax.axvline(config.max_review_share * 100, linestyle="--", color="#a34c36", label="Validation review cap")
    ax.set(xlabel="Applications flagged on validation (%)", ylabel="Defaults captured on validation (%)",
           title="Review capacity limits achievable recall", xlim=(0, 100), ylim=(0, 100))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures/review_tradeoff.png", dpi=160)
    plt.close(fig)

    matrix = np.array([[policy["tn"], policy["fp"]], [policy["fn"], policy["tp"]]])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{value:,}", ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Below threshold", "Flagged"],
           yticklabels=["Non-default", "Default"], xlabel="Predicted routing", ylabel="Observed label",
           title=f"Test outcomes at frozen threshold {policy['threshold']:.3f}")
    fig.tight_layout()
    fig.savefig(output / "figures/confusion_matrix.png", dpi=160)
    plt.close(fig)


def run_experiment(data: Path, output: Path, config: Config = Config()) -> dict:
    output = Path(output)
    for folder in ["figures", "tables"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    raw = load_data(Path(data), config)
    train, validation, test = split_data(raw, config)
    from eda import run_eda
    exploration = run_eda(train, output)
    x_train = engineer_features(train)
    encoder = preprocess(x_train, config.seed)
    # fit_transform cross-fits target encoding within training. transform uses
    # training-only mappings for both validation and test; unseen levels use the prior.
    train_encoded = encoder.fit_transform(x_train, train[TARGET])
    validation_encoded = encoder.transform(engineer_features(validation))
    models = build_models(train[TARGET], config)
    comparison = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(train_encoded, train[TARGET])
        scores = model.predict_proba(validation_encoded)[:, 1]
        comparison.append({"model": name,
                           "validation_roc_auc": float(roc_auc_score(validation[TARGET], scores)),
                           "validation_average_precision": float(average_precision_score(validation[TARGET], scores)),
                           "fit_seconds": time.perf_counter() - start})
        print(f"{name}: validation AUC {comparison[-1]['validation_roc_auc']:.4f}", flush=True)
    comparison = pd.DataFrame(comparison).sort_values(
        ["validation_roc_auc", "model"], ascending=[False, True]).reset_index(drop=True)
    selected_name = str(comparison.iloc[0].model)
    selected = models[selected_name]
    validation_scores = selected.predict_proba(validation_encoded)[:, 1]
    # Cost scale is estimated from training only. Scores are uncalibrated because
    # class weighting changes their scale; thresholds are empirical routing cutoffs.
    average_loan = float(train.DISBURSED_AMOUNT.mean())
    cost_fn = config.loss_given_default * average_loan
    cost_fp = config.lost_margin * average_loan
    threshold, threshold_table = choose_threshold(validation[TARGET], validation_scores, config, cost_fn, cost_fp)
    extended = None
    if config.extended:
        from experiments import run_extended_experiments
        extended = run_extended_experiments(train, validation, x_train, train_encoded,
                                           validation_encoded, encoder, models, selected_name,
                                           threshold_table, output, config)
    # Freeze the model and policy before looking at the held-out test outcomes.
    test_encoded = encoder.transform(engineer_features(test))
    test_scores = selected.predict_proba(test_encoded)[:, 1]
    policy = evaluate(test[TARGET], test_scores, threshold, cost_fn, cost_fp)
    standard = evaluate(test[TARGET], test_scores, 0.5, cost_fn, cost_fp)
    validation_policy = evaluate(validation[TARGET], validation_scores, threshold, cost_fn, cost_fp)
    baseline = evaluate(test[TARGET], np.full(len(test), train[TARGET].mean()), 1.0, cost_fn, cost_fp)
    from operations import run_operations
    operations = run_operations(train, validation, test, validation_scores, test_scores,
                                threshold, output, config)
    segments = pd.DataFrame({
        "segment": np.where(test_scores >= threshold, "Flagged for review", "Below review threshold"),
        "default": test[TARGET].to_numpy(),
    }).groupby("segment").agg(applications=("default", "size"), default_rate=("default", "mean"))
    segments["application_share"] = segments.applications / len(test)
    segments.to_csv(output / "tables/test_segments.csv")
    comparison.to_csv(output / "tables/model_comparison.csv", index=False)
    threshold_table.to_csv(output / "tables/validation_thresholds.csv", index=False)
    feature_names = encoder.get_feature_names_out()
    if hasattr(selected, "feature_importances_"):
        importance = pd.DataFrame({"feature": feature_names, "importance": selected.feature_importances_})
        importance["importance_type"] = "native model importance; not causal; scale depends on estimator"
    else:
        importance = pd.DataFrame({"feature": feature_names, "importance": np.abs(selected.named_steps["model"].coef_[0])})
        importance["importance_type"] = "absolute coefficient after standardization; not causal"
    importance.sort_values("importance", ascending=False).to_csv(output / "tables/feature_importance.csv", index=False)
    versions = {name: importlib.metadata.version(name) for name in [
        "numpy", "pandas", "scikit-learn", "matplotlib", "xgboost", "lightgbm", "catboost",
    ]}
    summary = {
        "run_type": "sample" if config.sample else "full_dataset",
        "config": asdict(config), "python": platform.python_version(), "versions": versions,
        "data_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        "rows_analyzed": len(raw), "default_count": int(raw[TARGET].sum()),
        "default_rate": float(raw[TARGET].mean()),
        "splits": {name: {"rows": len(part), "default_rate": float(part[TARGET].mean())}
                   for name, part in [("train", train), ("validation", validation), ("test", test)]},
        "selected_model": selected_name, "feature_count": int(train_encoded.shape[1]),
        "selection_metric": "validation ROC-AUC; no test-based model selection",
        "validation_policy": validation_policy, "test_policy": policy,
        "test_at_050": standard, "test_approve_all": baseline,
        "cost_assumptions": {"average_training_loan": average_loan, "cost_fn": cost_fn, "cost_fp": cost_fp,
                             "interpretation": "illustrative cost units; no verified realized savings"},
        "exploration": exploration,
        "extended_experiments": extended,
        "operations": operations,
        "limitations": [
            "Random application split; no out-of-time or unseen-entity validation.",
            "Validation set reused for model selection and policy choice; selection uncertainty remains.",
            "Class-weighted scores are uncalibrated and are not probability-of-default estimates.",
            "Review cap is fitted on validation; test or future flag rates can exceed it.",
            "Cost scenario assumes every flagged default loss is avoided and charges margin only for flagged non-defaults.",
            "Review does not itself prevent default; workload costs and actual interventions are not modeled.",
            "Geographic, staff, age, and identity-document features need fairness and availability review.",
            "Previously explored public dataset; held-out results are an internal benchmark, not external validation.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_results(comparison, test[TARGET], test_scores, policy, threshold_table, output, config)
    print(f"Selected {selected_name}; test AUC {policy['roc_auc']:.4f}; threshold {threshold:.3f}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--sample", type=int, help="Stratified development sample; do not present as a full-data run.")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="Skip extended diagnostics; keep the five-model benchmark and EDA.")
    args = parser.parse_args()
    published_reports = Path(__file__).resolve().parents[1] / "reports"
    if (args.sample or args.quick) and args.output.resolve() in {Path("reports").resolve(), published_reports.resolve()}:
        parser.error("Use --output reports/smoke for sample or quick runs to preserve complete results.")
    run_experiment(args.data, args.output, Config(sample=args.sample, jobs=args.jobs, extended=not args.quick))


if __name__ == "__main__":
    main()
