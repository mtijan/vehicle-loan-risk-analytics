"""Executable routing and monitoring demonstrations; no production integration."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from credit_risk import CATEGORICAL, TARGET, engineer_features
from eda import rate_table, plot_rates


def route_scores(scores, low, high):
    if not 0 <= low <= high:
        raise ValueError("Routing thresholds must satisfy 0 <= low <= high.")
    scores = np.asarray(scores, float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Routing requires finite scores between zero and one.")
    return np.where(scores >= high, "Priority review",
                    np.where(scores < low, "Lower-score queue", "Standard review"))


def choose_low_threshold(y, scores, high, maximum_rate=.10, minimum_share=.05):
    y, scores = np.asarray(y), np.asarray(scores)
    minimum_count = max(30, int(np.ceil(minimum_share * len(y))))
    eligible = []
    for threshold in np.arange(.005, high, .005):
        mask = scores < threshold
        if mask.sum() >= minimum_count and y[mask].mean() <= maximum_rate:
            eligible.append(float(threshold))
    return max(eligible, default=0.0)


def population_stability_index(reference, current, bins=10):
    """PSI using reference quantile edges, explicit missing bin, and smoothed shares."""
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").replace([np.inf, -np.inf], np.nan)
    cur = pd.to_numeric(pd.Series(current), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if len(ref) == 0 or len(cur) == 0 or ref.notna().sum() == 0:
        raise ValueError("PSI needs nonempty populations and observed reference values.")
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) == 1:
        edges = np.array([-np.inf, edges[0] - .5, edges[0] + .5, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.r_[np.histogram(ref.dropna(), edges)[0], ref.isna().sum()]
    cur_counts = np.r_[np.histogram(cur.dropna(), edges)[0], cur.isna().sum()]
    p, q = ref_counts / len(ref) + 1e-6, cur_counts / len(cur) + 1e-6
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def bootstrap_intervals(y, scores, threshold, repetitions=200, seed=42):
    """Stratified row bootstrap for a frozen model; conditions on observed prevalence."""
    y, scores = np.asarray(y), np.asarray(scores)
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        indices = np.r_[rng.choice(negative, len(negative), replace=True),
                        rng.choice(positive, len(positive), replace=True)]
        labels, sampled_scores = y[indices], scores[indices]
        pred = sampled_scores >= threshold
        values.append({"roc_auc": roc_auc_score(labels, sampled_scores),
                       "average_precision": average_precision_score(labels, sampled_scores),
                       "recall": float(pred[labels == 1].mean()),
                       "precision": float(labels[pred].mean()) if pred.any() else 0.,
                       "review_share": float(pred.mean())})
    frame = pd.DataFrame(values)
    return pd.DataFrame({"metric": frame.columns, "ci_low": frame.quantile(.025).to_numpy(),
                         "ci_high": frame.quantile(.975).to_numpy(), "bootstrap_repetitions": repetitions})


def run_operations(train, validation, test, validation_scores, test_scores, high, output, config):
    low = choose_low_threshold(validation[TARGET], validation_scores, high)
    validation_routes = route_scores(validation_scores, low, high)
    test_routes = route_scores(test_scores, low, high)
    for name, labels, y in [("validation", validation_routes, validation[TARGET]),
                             ("test", test_routes, test[TARGET])]:
        groups = pd.Series(labels, index=y.index)
        table = rate_table(groups, y).reindex(["Lower-score queue", "Standard review", "Priority review"])
        table.to_csv(output / "tables" / f"{name}_routing.csv")
        if name == "test":
            plotted = table[table.applications.fillna(0) > 0]
            plot_rates(plotted, "Frozen routing policy on test applications (95% Wilson intervals)",
                       output, "risk_routing", "Queue", horizontal=False)

    training_features, validation_features = engineer_features(train), engineer_features(validation)
    monitors = []
    for feature in ["LTV", "DISBURSED_AMOUNT", "PERFORM_CNS_SCORE", "NO_OF_INQUIRIES", "AGE_AT_DISBURSAL"]:
        reference, current = training_features[feature], validation_features[feature]
        monitors.append({"feature": feature, "psi": population_stability_index(reference, current),
                         "training_missing_share": float(reference.isna().mean()),
                         "validation_missing_share": float(current.isna().mean())})
    monitor_table = pd.DataFrame(monitors)
    monitor_table.to_csv(output / "tables/monitoring_feature_baseline.csv", index=False)
    unseen = pd.DataFrame([
        {"feature": col, "training_categories": training_features[col].nunique(),
         "validation_unseen_share": float((~validation_features[col].isin(set(training_features[col]))).mean())}
        for col in CATEGORICAL
    ])
    unseen.to_csv(output / "tables/monitoring_unseen_categories.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(monitor_table.feature, monitor_table.psi, color="#32658a")
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set(xlabel="PSI (training quantile reference bins)", xlim=(0, max(.01, monitor_table.psi.max() * 1.3)),
           title="Random-split population comparison: monitoring demonstration")
    fig.tight_layout()
    fig.savefig(output / "figures/monitoring_baseline.png", dpi=150)
    plt.close(fig)
    bootstrap = bootstrap_intervals(test[TARGET], test_scores, high, repetitions=200, seed=config.seed)
    bootstrap.to_csv(output / "tables/test_bootstrap_intervals.csv", index=False)
    monitoring_plan = pd.DataFrame([
        {"signal": "Schema, missingness, invalid values", "cadence": "Every scoring batch", "response": "Quarantine malformed inputs; inspect upstream changes"},
        {"signal": "Unseen sourcing categories and PSI", "cadence": "Compare each batch with training reference", "response": "Investigate population changes; thresholds require prospective calibration"},
        {"signal": "Flagged share vs review capacity", "cadence": "Daily queue review", "response": "Escalate capacity shortfalls; do not silently change the model cutoff"},
        {"signal": "ROC-AUC, AP, recall, calibration, subgroup outcomes", "cadence": "Once outcome labels mature", "response": "Compare with frozen reference; investigate changes before retraining"},
        {"signal": "Human overrides, review costs, intervention results", "cadence": "Pilot review cycle", "response": "Replace illustrative costs with observed operational evidence"},
        {"signal": "Versioned challenger evaluation", "cadence": "When data or policy changes justify a review", "response": "Validate on a new future-period holdout and retain a rollback version"},
    ])
    monitoring_plan.to_csv(output / "tables/monitoring_plan.csv", index=False)
    result = {
        "status": "offline routing and monitoring demonstration; integration remains a design",
        "low_score_threshold": low, "high_score_threshold": high,
        "low_queue_validation_max_observed_default_rate": .10,
        "low_queue_validation_min_share": .05,
        "routing_actions": {"Lower-score queue": "Normal underwriting; no automatic approval",
                            "Standard review": "Standard assessment with supporting evidence",
                            "Priority review": "Prioritize human review; no automatic rejection"},
        "monitoring_scope": "training reference vs random validation split; no live drift or future-period evidence",
        "bootstrap_repetitions": 200,
        "bootstrap_scope": "pointwise stratified row intervals, conditional on observed prevalence and frozen model; no fitting uncertainty",
    }
    (output / "routing_policy.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
