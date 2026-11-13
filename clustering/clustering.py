"""K-Means segmentation using input features only (no Churn label in fitting)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,davies_bouldin_score

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT / "results"
FIGURES=ROOT / "figures" / "clustering"
SEED=30


def main() -> None:
    RESULTS.mkdir(parents=True,exist_ok=True)
    FIGURES.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style="whitegrid")
    raw=pd.read_csv(ROOT / "data" / "CustomerChurn.csv")
    df=raw.drop_duplicates().copy().reset_index(drop=True)
    # The churn label is excluded from both scaling and model fitting.
    X=df.drop(columns="Churn")
    prep=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    X_scaled=prep.fit_transform(X)
    scores=[]
    for k in range(2,13):
        estimator=KMeans(n_clusters=k,random_state=SEED,n_init=10)
        labels=estimator.fit_predict(X_scaled)
        # Sampled silhouette keeps runtime reasonable; repeatable via fixed seed.
        sil=silhouette_score(X_scaled,labels,sample_size=min(1000,len(X_scaled)),random_state=SEED)
        db=davies_bouldin_score(X_scaled,labels)
        scores.append({"k":k,"silhouette_sample_1000":float(sil),"davies_bouldin":float(db),
                       "inertia":float(estimator.inertia_)})
    score_table=pd.DataFrame(scores)
    score_table.to_csv(RESULTS / "clustering_k_scores.csv",index=False,float_format="%.6f")
    best_k=int(score_table.sort_values(["silhouette_sample_1000","davies_bouldin"],ascending=[False,True]).iloc[0]["k"])
    best=KMeans(n_clusters=best_k,random_state=SEED,n_init=10)
    labels=best.fit_predict(X_scaled)
    profile=df.assign(cluster=labels).groupby("cluster").agg(
        count=("Churn","size"),
        churn_rate=("Churn","mean"),
        average_customer_value=("Customer Value","mean"),
        average_frequency_of_use=("Frequency of use","mean"),
        average_subscription_length=("Subscription  Length","mean")
    ).reset_index()
    profile.to_csv(RESULTS / "clustering_cluster_profile.csv",index=False,float_format="%.6f")
    summary={"original_rows":int(len(raw)),"removed_duplicate_rows":int(len(raw)-len(df)),
        "rows_used":int(len(df)),"k_values_tested":list(range(2,13)),
        "k_selection":"maximum silhouette from a reproducible 1000-observation sample",
        "silhouette_is_sampled":True,"silhouette_sample_size":min(1000,len(df)),
        "best_k":best_k,"best_silhouette_sample_1000":float(score_table.loc[score_table.k==best_k,"silhouette_sample_1000"].iloc[0]),
        "best_davies_bouldin":float(score_table.loc[score_table.k==best_k,"davies_bouldin"].iloc[0]),
        "target_used_for_fitting":False,"churn_used_only_for_post_hoc_profiling":True}
    (RESULTS / "clustering_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")

    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].plot(score_table.k,score_table.silhouette_sample_1000,marker="o")
    axs[0].set(xlabel="Number of clusters (k)",ylabel="Silhouette (sampled)",title="Higher is better")
    axs[1].plot(score_table.k,score_table.davies_bouldin,marker="o")
    axs[1].set(xlabel="Number of clusters (k)",ylabel="Davies–Bouldin",title="Lower is better")
    fig.tight_layout();fig.savefig(FIGURES / "k_selection_metrics.png",dpi=155);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(profile.cluster.astype(str),profile["count"])
    ax.set(xlabel="Cluster",ylabel="Customers",title=f"Cluster sizes (k={best_k})")
    fig.tight_layout();fig.savefig(FIGURES / "cluster_sizes.png",dpi=155);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(profile.cluster.astype(str),100*profile.churn_rate)
    ax.set(xlabel="Cluster",ylabel="Observed churn rate (%)",title="Post-hoc churn distribution by cluster")
    fig.tight_layout();fig.savefig(FIGURES / "cluster_churn_rates.png",dpi=155);plt.close(fig)

    pca=PCA(n_components=2,random_state=SEED)
    embedded=pca.fit_transform(X_scaled)
    fig,ax=plt.subplots(figsize=(8,6))
    scatter=ax.scatter(embedded[:,0],embedded[:,1],c=labels,cmap="tab10",alpha=.6,s=10)
    fig.colorbar(scatter,ax=ax,label="Cluster")
    ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",title="K-Means clusters in PCA projection")
    fig.tight_layout();fig.savefig(FIGURES / "pca_cluster_projection.png",dpi=155);plt.close(fig)

    print("Clustering complete. Selected k:",best_k)
    print("Silhouette (sample):",round(summary["best_silhouette_sample_1000"],4))


if __name__=="__main__":
    main()
