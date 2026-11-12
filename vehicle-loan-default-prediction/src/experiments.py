"""Validation-only model diagnostics; no test outcomes are accepted here."""
from dataclasses import replace
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from credit_risk import TARGET, build_models, engineer_features, preprocess

ENTITY_COLUMNS = ["SUPPLIER_ID", "BRANCH_ID", "STATE_ID", "MANUFACTURER_ID", "CURRENT_PINCODE_ID", "EMPLOYEE_CODE_ID"]


def diagnostic_row(name, y, scores):
    return {"experiment": name, "validation_roc_auc": float(roc_auc_score(y, scores)),
            "validation_average_precision": float(average_precision_score(y, scores))}


def cost_sensitivity(grid, average_loan):
    """Reuse validation confusion counts; no repeated model fitting or test tuning."""
    rows = []
    for capacity in [.10, .20, .25, .35, .50]:
        for lgd in [.25, .45, .65]:
            for margin in [.04, .08, .12]:
                eligible = grid.loc[grid.review_share <= capacity].copy()
                eligible["scenario_cost"] = (eligible.fn * lgd + eligible.fp * margin) * average_loan
                chosen = eligible.sort_values(["scenario_cost", "threshold"], ascending=[True, False]).iloc[0]
                baseline = (chosen.tp + chosen.fn) * lgd * average_loan
                rows.append({"review_cap": capacity, "lgd_assumption": lgd, "margin_assumption": margin,
                             "threshold": float(chosen.threshold), "review_share": float(chosen.review_share),
                             "recall": float(chosen.recall), "precision": float(chosen.precision),
                             "scenario_cost_reduction": float(1 - chosen.scenario_cost / baseline)})
    return pd.DataFrame(rows)


def run_extended_experiments(train, validation, x_train, train_encoded, validation_encoded,
                             encoder, models, selected_name, grid, output, config):
    selected = models[selected_name]
    y_train, y_validation = train[TARGET], validation[TARGET]
    val_scores = selected.predict_proba(validation_encoded)[:, 1]
    primary = diagnostic_row("Primary selected model", y_validation, val_scores)
    # Core selection remains the five predeclared model families. These challengers
    # diagnose sensitivity and are never promoted by inspecting the test set.
    variants = [primary]
    if selected_name == "Logistic Regression":
        unweighted = clone(selected).set_params(model__class_weight=None)
    elif selected_name == "XGBoost":
        unweighted = clone(selected).set_params(scale_pos_weight=1.0)
    elif selected_name == "CatBoost":
        unweighted = clone(selected).set_params(auto_class_weights=None)
    else:
        unweighted = clone(selected).set_params(class_weight=None)
    start = time.perf_counter()
    unweighted.fit(train_encoded, y_train)
    row = diagnostic_row("Without class weighting", y_validation, unweighted.predict_proba(validation_encoded)[:, 1])
    row["fit_seconds"] = time.perf_counter() - start
    variants.append(row)
    for name, columns in [
        ("Without sourcing and geographic IDs", [c for c in x_train if c not in ENTITY_COLUMNS]),
        ("Without engineered numeric features", [c for c in x_train if c in train.columns]),
    ]:
        transformed = preprocess(x_train[columns], config.seed)
        reduced_train = transformed.fit_transform(x_train[columns], y_train)
        reduced_validation = transformed.transform(engineer_features(validation)[columns])
        model = clone(selected)
        start = time.perf_counter()
        model.fit(reduced_train, y_train)
        row = diagnostic_row(name, y_validation, model.predict_proba(reduced_validation)[:, 1])
        row.update(feature_count=len(columns), fit_seconds=time.perf_counter() - start)
        variants.append(row)
    ablations = pd.DataFrame(variants)
    ablations["auc_change_from_primary"] = ablations.validation_roc_auc - primary["validation_roc_auc"]
    ablations.to_csv(output / "tables/ablation_experiments.csv", index=False)
    print("Extended: class weighting and feature ablations complete", flush=True)

    # Small explicit challenger grid on validation; keep the primary unchanged.
    challengers = [primary]
    if selected_name == "LightGBM":
        parameter_grid = [
            ("Smaller trees (15 leaves)", {"num_leaves": 15}),
            ("Stronger regularization", {"reg_lambda": 20, "min_child_samples": 200}),
        ]
    else:
        parameter_grid = []
    for name, params in parameter_grid:
        challenger = clone(selected).set_params(**params)
        challenger.fit(train_encoded, y_train)
        row = diagnostic_row(name, y_validation, challenger.predict_proba(validation_encoded)[:, 1])
        row["changed_parameters"] = json.dumps(params, sort_keys=True)
        challengers.append(row)
    pd.DataFrame(challengers).to_csv(output / "tables/hyperparameter_challengers.csv", index=False)

    ensemble = .35 * models["XGBoost"].predict_proba(validation_encoded)[:, 1]
    ensemble += .35 * models["LightGBM"].predict_proba(validation_encoded)[:, 1]
    ensemble += .30 * models["CatBoost"].predict_proba(validation_encoded)[:, 1]
    ensemble_rows = [primary, diagnostic_row("Fixed blend: XGB 0.35 / LGBM 0.35 / Cat 0.30", y_validation, ensemble)]
    pd.DataFrame(ensemble_rows).to_csv(output / "tables/ensemble_validation.csv", index=False)

    # Outer-fold validation rebuilds all preprocessing from each fold's training rows.
    # TargetEncoder performs its own inner cross-fitting inside these outer folds.
    fold_rows = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    for fold, (fit_index, holdout_index) in enumerate(folds.split(x_train, y_train), 1):
        fitted_encoder = preprocess(x_train.iloc[fit_index], config.seed)
        fit_x = fitted_encoder.fit_transform(x_train.iloc[fit_index], y_train.iloc[fit_index])
        holdout_x = fitted_encoder.transform(x_train.iloc[holdout_index])
        model = clone(selected)
        model.fit(fit_x, y_train.iloc[fit_index])
        scores = model.predict_proba(holdout_x)[:, 1]
        fold_rows.append({"fold": fold, "fit_rows": len(fit_index), "holdout_rows": len(holdout_index),
                          "roc_auc": float(roc_auc_score(y_train.iloc[holdout_index], scores)),
                          "average_precision": float(average_precision_score(y_train.iloc[holdout_index], scores))})
        print(f"Extended: outer CV fold {fold}/5 AUC {fold_rows[-1]['roc_auc']:.4f}", flush=True)
    cv = pd.DataFrame(fold_rows)
    cv.to_csv(output / "tables/cross_validation.csv", index=False)

    seeds = []
    for seed in [config.seed, 7, 123, 2024]:
        if seed == config.seed:
            score = val_scores
        else:
            seeded = build_models(y_train, replace(config, seed=seed))[selected_name]
            seeded.fit(train_encoded, y_train)
            score = seeded.predict_proba(validation_encoded)[:, 1]
        seeds.append({"model_seed": seed, "validation_roc_auc": float(roc_auc_score(y_validation, score)),
                      "validation_average_precision": float(average_precision_score(y_validation, score))})
    pd.DataFrame(seeds).to_csv(output / "tables/seed_stability.csv", index=False)
    print("Extended: seed stability complete", flush=True)

    sensitivity = cost_sensitivity(grid, float(train.DISBURSED_AMOUNT.mean()))
    sensitivity.to_csv(output / "tables/cost_capacity_sensitivity.csv", index=False)
    # In-sample training fit is descriptive, not an unbiased performance estimate.
    fit_scores = selected.predict_proba(train_encoded)[:, 1]
    generalization = pd.DataFrame([
        {"partition": "In-sample cross-fitted training representation", "roc_auc": roc_auc_score(y_train, fit_scores),
         "average_precision": average_precision_score(y_train, fit_scores)},
        {"partition": "Validation", "roc_auc": primary["validation_roc_auc"],
         "average_precision": primary["validation_average_precision"]},
    ])
    generalization.to_csv(output / "tables/generalization_gap.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].barh(ablations.experiment, ablations.validation_roc_auc, color="#32658a")
    axes[0].set(xlim=(0, 1), xlabel="Validation ROC-AUC", title="Feature and weighting ablations")
    axes[1].plot(cv.fold, cv.roc_auc, "o-", color="#32658a")
    axes[1].axhline(cv.roc_auc.mean(), linestyle="--", color="#777777", label="Fold mean")
    axes[1].set(xticks=range(1, 6), ylim=(.5, 1), xlabel="Outer fold (training partition only)",
                ylabel="ROC-AUC", title="Five-fold preprocessing and model validation")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "figures/experiment_diagnostics.png", dpi=150)
    plt.close(fig)
    baseline_costs = sensitivity.loc[np.isclose(sensitivity.lgd_assumption, .45) & np.isclose(sensitivity.margin_assumption, .08)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(baseline_costs.review_cap * 100, baseline_costs.recall * 100, "o-", label="Validation recall")
    ax.plot(baseline_costs.review_cap * 100, baseline_costs.review_share * 100, "s--", label="Validation flagged share")
    ax.set(xlabel="Allowed validation review capacity (%)", ylabel="Percent (%)", xlim=(0, 55), ylim=(0, 100),
           title="Review capacity changes detection under fixed cost assumptions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures/capacity_sensitivity.png", dpi=150)
    plt.close(fig)

    result = {
        "scope": "training and validation only; challengers do not replace the primary family selection",
        "ablation_count": len(ablations), "challenger_count": len(challengers),
        "cv_folds": len(cv), "cv_auc_mean": float(cv.roc_auc.mean()),
        "cv_auc_std": float(cv.roc_auc.std(ddof=1)),
        "model_seeds": [r["model_seed"] for r in seeds],
        "seed_validation_auc_range": float(max(r["validation_roc_auc"] for r in seeds) - min(r["validation_roc_auc"] for r in seeds)),
        "sensitivity_scenarios": len(sensitivity),
        "primary_training_auc": float(roc_auc_score(y_train, fit_scores)),
        "blend_validation_auc": float(roc_auc_score(y_validation, ensemble)),
        "notes": ["Outer-fold evaluation rebuilds imputation and target encoding in every fold.",
                  "The model family was selected using a separate validation set before its training-only CV diagnostics.",
                  "Seed experiments change estimator randomness while holding data split and encoding fixed.",
                  "A small challenger grid is exploratory and not an exhaustive hyperparameter search."],
    }
    (output / "experiments.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
