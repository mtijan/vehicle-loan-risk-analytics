"""Training-only exploratory analysis. Export aggregates, never application rows."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#32658a"
ORANGE = "#a34f35"
TARGET = "LOAN_DEFAULT"


def wilson_interval(events, count, z=1.959963984540054):
    """Wilson binomial interval; undefined empty groups return NaN."""
    events, count = np.asarray(events, float), np.asarray(count, float)
    safe = np.where(count > 0, count, np.nan)
    rate = events / safe
    denominator = 1 + z * z / safe
    center = (rate + z * z / (2 * safe)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / safe + z * z / (4 * safe * safe)) / denominator
    return np.maximum(0, center - half), np.minimum(1, center + half)


def rate_table(labels, y):
    grouped = pd.DataFrame({"group": labels, "outcome": y}).groupby("group", observed=True, sort=True)
    result = grouped.outcome.agg(applications="size", defaults="sum", default_rate="mean")
    result["ci_low"], result["ci_high"] = wilson_interval(result.defaults, result.applications)
    result["application_share"] = result.applications / len(y)
    return result


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / "figures" / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rates(table, title, output, name, xlabel, horizontal=False):
    fig, ax = plt.subplots(figsize=(9, max(4.5, len(table) * 0.34) if horizontal else 5))
    positions = np.arange(len(table))
    errors = np.vstack([table.default_rate - table.ci_low, table.ci_high - table.default_rate]) * 100
    labels = [str(v) for v in table.index]
    values = table.default_rate.to_numpy() * 100
    if horizontal:
        ax.barh(positions, values, xerr=errors, color=BLUE, capsize=3)
        ax.set(yticks=positions, yticklabels=labels, xlabel="Observed default rate (%)", ylabel=xlabel,
               xlim=(0, max(table.ci_high) * 122), title=title)
        for i, (v, count) in enumerate(zip(values, table.applications)):
            ax.text(table.ci_high.iloc[i] * 100 + 0.5, i, f"{v:.1f}% | n={count:,}", va="center", fontsize=8)
    else:
        ax.bar(positions, values, yerr=errors, color=BLUE, capsize=3)
        ax.set(xticks=positions, xticklabels=labels, ylabel="Observed default rate (%)", xlabel=xlabel,
               ylim=(0, max(table.ci_high) * 130), title=title)
        for i, (v, count) in enumerate(zip(values, table.applications)):
            ax.text(i, table.ci_high.iloc[i] * 100 + 0.6, f"{v:.1f}%\nn={count:,}", ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, output, name)


def run_eda(train: pd.DataFrame, output: Path) -> dict:
    output = Path(output)
    for folder in ["figures", "tables"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    from credit_risk import engineer_features
    engineered = engineer_features(train)
    y = train[TARGET]
    chart_names = []

    def rates(labels, name, title, xlabel, horizontal=False, minimum=0):
        table = rate_table(labels, y)
        table.to_csv(output / "tables" / f"{name}.csv")
        plotted = table[table.applications >= minimum]
        if not plotted.empty:
            plot_rates(plotted, title, output, name, xlabel, horizontal)
            chart_names.append(name)
        return table

    quality = pd.DataFrame({"column": train.columns, "dtype": train.dtypes.astype(str).to_numpy(),
                            "missing_count": train.isna().sum().to_numpy(),
                            "missing_share": train.isna().mean().to_numpy(),
                            "unique_values": train.nunique(dropna=True).to_numpy()})
    quality.to_csv(output / "tables/data_quality.csv", index=False)
    missing = quality.set_index("column").missing_share.sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(missing.index[::-1], missing.to_numpy()[::-1] * 100, color=BLUE)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.set(xlabel="Missing values (%)", xlim=(0, max(1, missing.max() * 130)),
           title="Training data completeness: 12 columns with the highest missing share")
    save(fig, output, "missingness")
    chart_names.append("missingness")

    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Non-default", "Default"], counts, color=[BLUE, ORANGE])
    ax.bar_label(bars, labels=[f"{n:,}\n{n / len(train):.2%}" for n in counts], padding=4)
    ax.set(ylabel="Applications", ylim=(0, counts.max() * 1.2), title="Training outcome distribution")
    save(fig, output, "target_distribution")
    chart_names.append("target_distribution")

    financial_cols = ["DISBURSED_AMOUNT", "ASSET_COST", "LTV", "PRI_CURRENT_BALANCE"]
    train[financial_cols].describe(percentiles=[.01, .25, .5, .75, .95, .99]).T.to_csv(output / "tables/financial_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.flat, financial_cols):
        values = train[col].dropna()
        high = values.quantile(.99)
        display_values = values.clip(upper=high)
        ax.hist(display_values, bins=40, color=BLUE, edgecolor="white")
        ax.set(title=f"{col} (upper display clipped at p99)", xlabel="Percent" if col == "LTV" else "Dataset amount units",
               ylabel="Applications", yscale="log")
    save(fig, output, "financial_distributions")
    chart_names.append("financial_distributions")
    quantile_labels = pd.qcut(train.DISBURSED_AMOUNT, 5, duplicates="drop")
    rates(quantile_labels, "loan_amount_default_rates", "Default rate by loan-size quintile (95% Wilson intervals)",
          "Loan amount range (dataset units)")

    ltv = rates(pd.cut(train.LTV, [0, 60, 70, 80, 90, 100], include_lowest=True,
                       labels=["0-60", "60-70", "70-80", "80-90", "90-100"]),
                "ltv_default_rates", "Default rates by LTV band (95% Wilson intervals)", "Loan-to-value band (%)")
    rates(pd.Series(np.where(train.PERFORM_CNS_SCORE.eq(0), "Score equals zero", "Score above zero"), index=train.index),
          "bureau_score_presence", "Zero and nonzero bureau scores are different populations", "Bureau-score group")
    bureau = rates(train.PERFORM_CNS_SCORE_DESCRIPTION.fillna("Unknown"), "bureau_categories",
                   "Default rate by bureau description (n >= 500; 95% intervals)", "Bureau description", True, minimum=500)
    rates(train.EMPLOYMENT_TYPE.fillna("Unknown"), "employment_default_rates",
          "Employment groups: uncertainty and sample size matter", "Employment type")
    for column, name in [("NO_OF_INQUIRIES", "inquiries_default_rates"),
                          ("DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS", "delinquency_default_rates")]:
        labels = pd.Series(np.where(train[column] >= 3, "3+", train[column].astype(str)), index=train.index)
        rates(labels, name, f"Default by {column.lower().replace('_', ' ')}", "Count (3 or more grouped together)")
    age_groups = pd.cut(engineered.AGE_AT_DISBURSAL, [18, 25, 35, 45, 55, 100], include_lowest=True,
                        labels=["18-25", "25-35", "35-45", "45-55", "55-100"])
    rates(age_groups, "age_default_rates", "Age groups describe associations, not lending eligibility", "Age at disbursal (years)")
    dates = pd.to_datetime(train.DISBURSAL_DATE, format="%d-%m-%Y", errors="coerce")
    monthly = rates(dates.dt.to_period("M").astype(str), "monthly_default_rates",
                    "Default rates by disbursal month (training only)", "Disbursal month")

    context = []
    for column in ["SUPPLIER_ID", "BRANCH_ID", "CURRENT_PINCODE_ID", "EMPLOYEE_CODE_ID"]:
        group = train.groupby(column)[TARGET].agg(["size", "mean"])
        volume = group["size"].sort_values(ascending=False)
        context.append({"entity_type": column, "unique_entities": len(group),
                        "top_10_volume_share": float(volume.head(10).sum() / len(train)),
                        "entities_with_fewer_than_30_applications": int((group["size"] < 30).sum()),
                        "applications_in_small_entities_share": float(group.loc[group["size"] < 30, "size"].sum() / len(train))})
    pd.DataFrame(context).to_csv(output / "tables/entity_concentration.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([c["entity_type"].replace("_ID", "") for c in context], [c["top_10_volume_share"] * 100 for c in context], color=BLUE)
    ax.set(ylabel="Applications in 10 largest entities (%)", ylim=(0, 100), title="Sourcing concentration can affect model transfer")
    save(fig, output, "entity_concentration")
    chart_names.append("entity_concentration")

    corr_cols = ["DISBURSED_AMOUNT", "ASSET_COST", "LTV", "PERFORM_CNS_SCORE", "PRI_ACTIVE_ACCTS",
                 "PRI_OVERDUE_ACCTS", "NO_OF_INQUIRIES", "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS", TARGET]
    corr = train[corr_cols].corr()
    corr.to_csv(output / "tables/financial_correlations.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    short = [c.replace("DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS", "RECENT_DELINQUENCY") for c in corr_cols]
    ax.set(xticks=range(len(short)), yticks=range(len(short)), xticklabels=short, yticklabels=short,
           title="Training Pearson correlations: descriptive linear associations")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    for (i, j), val in np.ndenumerate(corr.to_numpy()):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(val) > .6 else "black")
    fig.colorbar(plot, ax=ax, label="Pearson r")
    save(fig, output, "correlation_matrix")
    chart_names.append("correlation_matrix")

    score_band = pd.Series(np.where(train.PERFORM_CNS_SCORE.eq(0), "Score = 0",
                                   np.where(train.PERFORM_CNS_SCORE < 650, "1-649", "650+")), index=train.index)
    cross = pd.DataFrame({"ltv": pd.cut(train.LTV, [0, 70, 80, 100], labels=["<=70", "70-80", ">80"]),
                          "bureau": score_band, "outcome": y}).groupby(["ltv", "bureau"], observed=True).outcome.agg(["size", "mean"])
    cross.to_csv(output / "tables/ltv_bureau_interaction.csv")
    values = cross["mean"].unstack().reindex(index=["<=70", "70-80", ">80"], columns=["Score = 0", "1-649", "650+"])
    samples = cross["size"].unstack().reindex_like(values)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(values * 100, cmap="Blues", vmin=0, vmax=max(40, np.nanmax(values.to_numpy()) * 100))
    for (i, j), val in np.ndenumerate(values.to_numpy()):
        count = samples.iloc[i, j]
        ax.text(j, i, f"{val:.1%}\nn={count:,.0f}" if np.isfinite(val) else "No data", ha="center", va="center")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=values.columns, yticklabels=values.index,
           xlabel="Bureau-score group", ylabel="LTV band (%)", title="Loan structure and bureau score jointly describe risk")
    fig.colorbar(image, ax=ax, label="Observed default rate (%)")
    save(fig, output, "ltv_bureau_interaction")
    chart_names.append("ltv_bureau_interaction")

    summary = {
        "partition": "training only", "training_rows": len(train), "chart_count": len(chart_names),
        "charts": chart_names, "duplicate_application_ids": int(train.UNIQUEID.duplicated().sum()),
        "duplicate_full_rows": int(train.duplicated().sum()),
        "training_ltv_above_80_default_rate": float(y[train.LTV > 80].mean()),
        "training_ltv_at_most_80_default_rate": float(y[train.LTV <= 80].mean()),
        "training_zero_bureau_score_share": float(train.PERFORM_CNS_SCORE.eq(0).mean()),
        "invalid_age_count": int(engineered.AGE_AT_DISBURSAL.isna().sum()),
        "disbursal_start": str(dates.min().date()), "disbursal_end": str(dates.max().date()),
        "notes": ["Rate intervals are pointwise 95% Wilson intervals, not multiple-comparison adjusted.",
                  "Financial histograms clip display values at p99 and use logarithmic count axes; model inputs are not clipped.",
                  "Monthly training comparisons are not out-of-time model validation.",
                  "Bureau descriptions with fewer than 500 applications remain in CSV but are omitted from the chart."],
    }
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
