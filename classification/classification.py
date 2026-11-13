"""Supervised churn classification: KNN and Decision Tree with leakage-aware CV.

Original assignment code: original_code/classification_original.py
Standalone: python classification/classification.py
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "classification"
MODELS = ROOT / "models"
SEED = 42


def candidates():
    models = []
    for k in (5,7,13):
        models.append((f"KNN (k={k})", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=k)),
        ])))
    for depth in (3,5,8):
        models.append((f"Decision Tree (max_depth={depth})", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(max_depth=depth,random_state=SEED)),
        ])))
    return models


def main() -> None:
    FIGURES.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    MODELS.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style="whitegrid")
    raw=pd.read_csv(ROOT / "data" / "CustomerChurn.csv")
    data=raw.drop_duplicates().copy()  # Prevent identical records from crossing the split.
    X=data.drop(columns="Churn")
    y=data["Churn"].astype(int)
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,stratify=y,test_size=0.30,random_state=SEED)

    # A fitted scaler is created *within* CV folds and later fitted only to training data.
    cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=SEED)
    rows=[]
    for name,model in candidates():
        scores=cross_validate(model,X_train,y_train,cv=cv,
            scoring={"f1":"f1","precision":"precision","recall":"recall","accuracy":"accuracy"},
            n_jobs=1, error_score="raise")
        rows.append({"model":name,
            "cv_f1_mean":float(scores["test_f1"].mean()),
            "cv_f1_std":float(scores["test_f1"].std()),
            "cv_precision_mean":float(scores["test_precision"].mean()),
            "cv_recall_mean":float(scores["test_recall"].mean()),
            "cv_accuracy_mean":float(scores["test_accuracy"].mean())})
    leaderboard=pd.DataFrame(rows).sort_values(["cv_f1_mean","cv_recall_mean"],ascending=False)
    leaderboard.to_csv(RESULTS / "classification_cv_results.csv",index=False,float_format="%.6f")

    # Choose via training-only cross-validation; touch the hold-out set once.
    best_name=leaderboard.iloc[0]["model"]
    model=dict(candidates())[best_name]
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    prob=model.predict_proba(X_test)[:,list(model.classes_).index(1)]
    cm=confusion_matrix(y_test,pred,labels=[0,1])
    pd.DataFrame(cm,index=["actual_0","actual_1"],columns=["predicted_0","predicted_1"]).to_csv(
        RESULTS / "classification_confusion_matrix.csv")
    result={
        "selection_criterion":"maximum 10-fold training CV F1 for churn=1; tie-break by recall",
        "best_model":best_name, "seed":SEED,
        "original_rows":int(len(raw)), "removed_duplicate_rows":int(len(raw)-len(data)),
        "rows_after_dedup":int(len(data)),
        "training_rows":int(len(X_train)), "test_rows":int(len(X_test)),
        "train_churn_count":int(y_train.sum()), "test_churn_count":int(y_test.sum()),
        "test_accuracy":float(accuracy_score(y_test,pred)),
        "test_precision_churn":float(precision_score(y_test,pred,zero_division=0)),
        "test_recall_churn":float(recall_score(y_test,pred,zero_division=0)),
        "test_f1_churn":float(f1_score(y_test,pred,zero_division=0)),
        "test_roc_auc":float(roc_auc_score(y_test,prob)),
        "test_average_precision":float(average_precision_score(y_test,prob)),
        "confusion_matrix_labels":[0,1],"confusion_matrix":cm.tolist()
    }
    (RESULTS / "classification_test_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    (RESULTS / "classification_test_report.txt").write_text(
        "Selected: "+best_name+"\n"+classification_report(y_test,pred,digits=4,zero_division=0),
        encoding="utf8")
    joblib.dump(model,MODELS / "best_classifier.joblib")

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.barh(leaderboard["model"][::-1],leaderboard["cv_f1_mean"][::-1])
    ax.set(xlabel="10-fold CV F1 (churn class)",title="Model selection on training data",xlim=(0,1))
    fig.tight_layout();fig.savefig(FIGURES / "model_comparison_cv_f1.png",dpi=155);plt.close(fig)

    fig,ax=plt.subplots(figsize=(5.5,4.5))
    ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["No churn","Churn"]).plot(ax=ax,cmap="Blues",colorbar=False)
    ax.set_title("Holdout confusion matrix")
    fig.tight_layout();fig.savefig(FIGURES / "confusion_matrix.png",dpi=155);plt.close(fig)

    fpr,tpr,_=roc_curve(y_test,prob)
    fig,ax=plt.subplots(figsize=(6,4.8))
    ax.plot(fpr,tpr,label=f'ROC AUC = {result["test_roc_auc"]:.3f}')
    ax.plot([0,1],[0,1],ls="--",color="gray")
    ax.set(xlabel="False positive rate",ylabel="True positive rate",title="ROC on holdout test")
    ax.legend();fig.tight_layout();fig.savefig(FIGURES / "roc_curve.png",dpi=155);plt.close(fig)

    precision,recall,_=precision_recall_curve(y_test,prob)
    fig,ax=plt.subplots(figsize=(6,4.8))
    ax.plot(recall,precision,label=f'Average precision = {result["test_average_precision"]:.3f}')
    ax.set(xlabel="Recall",ylabel="Precision",title="Precision–recall on holdout test")
    ax.legend();fig.tight_layout();fig.savefig(FIGURES / "precision_recall_curve.png",dpi=155);plt.close(fig)

    if best_name.startswith("Decision Tree"):
        importances=pd.Series(model.named_steps["classifier"].feature_importances_,index=X.columns).sort_values()
        fig,ax=plt.subplots(figsize=(8,6))
        importances.plot.barh(ax=ax)
        ax.set(xlabel="Gini importance",title="Selected decision-tree feature importances")
        fig.tight_layout();fig.savefig(FIGURES / "tree_feature_importance.png",dpi=155);plt.close(fig)

    print("Classification complete. Best:",best_name)
    print("Holdout churn F1:",f'{result["test_f1_churn"]:.4f}',"ROC AUC:",f'{result["test_roc_auc"]:.4f}')


if __name__ == "__main__":
    main()
