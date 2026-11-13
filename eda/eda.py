"""Reproducible EDA of the original Iranian Churn CSV.

Original assignment: original_code/eda_original.py
Standalone: python eda/eda.py
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "figures" / "eda"
RESULTS = ROOT / "results"


def export_chart(name: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=155, bbox_inches="tight")
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "CustomerChurn.csv")
    numeric = df.select_dtypes(include="number")
    report = pd.DataFrame(index=numeric.columns)
    report["count"] = numeric.count()
    report["mean"] = numeric.mean()
    report["median"] = numeric.median()
    report["mode"] = numeric.mode(dropna=True).iloc[0]
    report["variance_population"] = numeric.var(ddof=0)
    report["std_population"] = numeric.std(ddof=0)
    report.to_csv(RESULTS / "eda_descriptive_statistics.csv", index_label="feature", float_format="%.5f")
    counts = df["Churn"].value_counts().sort_index()
    overview = {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "predictor_features": int(df.shape[1]-1),
        "missing_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "target_counts": {str(k): int(v) for k,v in counts.items()},
        "churn_rate": float(df["Churn"].mean()),
    }
    (RESULTS / "dataset_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    fig,ax=plt.subplots(figsize=(6,4))
    sns.countplot(data=df,x="Churn",hue="Churn",legend=False,palette="Set2",ax=ax)
    ax.set(title="Churn class distribution",xticks=[0,1],xticklabels=["No churn (0)","Churn (1)"])
    export_chart("churn_distribution.png")

    fig,ax=plt.subplots(figsize=(8,4))
    sns.boxplot(data=df,x="Complains",y="Call  Failure",ax=ax)
    ax.set(title="Call failures by complaints",xlabel="Complaints",ylabel="Call failures")
    export_chart("complaints_vs_call_failures.png")

    fig,ax=plt.subplots(figsize=(8,4))
    sns.boxplot(data=df,x="Age",y="Subscription  Length",ax=ax)
    ax.set(title="Subscription length by age",ylabel="Subscription length")
    export_chart("subscription_length_by_age.png")

    for col,name in [("Frequency of use","frequency_of_use_boxplot.png"),
                     ("Frequency of SMS","frequency_of_sms_boxplot.png"),
                     ("Subscription  Length","subscription_length_boxplot.png")]:
        fig,ax=plt.subplots(figsize=(8,3.8))
        sns.boxplot(x=df[col],ax=ax)
        ax.set(title=col)
        export_chart(name)

    for col,name in [("Call  Failure","call_failures_histogram.png"),
                     ("Seconds of Use","seconds_of_use_histogram.png")]:
        fig,ax=plt.subplots(figsize=(8,4))
        sns.histplot(df[col],bins=20,kde=True,ax=ax,color="steelblue")
        ax.set(title=col)
        export_chart(name)

    fig,ax=plt.subplots(figsize=(11,9))
    sns.heatmap(numeric.corr(),cmap="vlag",center=0,ax=ax)
    ax.set_title("Feature correlation heatmap")
    export_chart("correlation_heatmap.png")
    print("EDA complete:",overview)
    print("Charts:",CHARTS)


if __name__ == "__main__":
    main()
