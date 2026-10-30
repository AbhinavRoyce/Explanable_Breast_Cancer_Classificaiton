from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, RocCurveDisplay
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "breast_cancer.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)  # Positive class: malignant
    print("=" * 72)
    print("EXPLAINABLE BREAST CANCER CLASSIFICATION: MODEL COMPARISON")
    print("=" * 72)
    print(f"Dataset shape: {df.shape}")
    print(f"Class counts (0=benign, 1=malignant):\n{y.value_counts().sort_index()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Gaussian Naive Bayes": GaussianNB(),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision_malignant": precision_score(y_test, pred, zero_division=0),
            "Recall_malignant": recall_score(y_test, pred, zero_division=0),
            "F1_malignant": f1_score(y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, prob),
            "5_fold_CV_accuracy_mean": cv_scores.mean(),
            "5_fold_CV_accuracy_std": cv_scores.std()
        })
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
    results.to_csv(OUT / "model_comparison.csv", index=False)
    print("\nTEST SET MODEL COMPARISON (positive class = malignant)")
    print(results.round(4).to_string(index=False))

    # Main contribution experiment: full 30 features vs 10 features selected on training data only.
    selector_model = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=10)),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ])
    selector_model.fit(X_train, y_train)
    sel_pred = selector_model.predict(X_test)
    sel_prob = selector_model.predict_proba(X_test)[:, 1]
    selected_mask = selector_model.named_steps["select"].get_support()
    selected_features = X.columns[selected_mask].tolist()

    baseline = fitted["Logistic Regression"]
    base_pred = baseline.predict(X_test)
    base_prob = baseline.predict_proba(X_test)[:, 1]
    ablation = pd.DataFrame([
        {
            "Experiment": "Logistic Regression with all 30 features",
            "Number_of_features": X.shape[1],
            "Accuracy": accuracy_score(y_test, base_pred),
            "Precision_malignant": precision_score(y_test, base_pred, zero_division=0),
            "Recall_malignant": recall_score(y_test, base_pred, zero_division=0),
            "F1_malignant": f1_score(y_test, base_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, base_prob)
        },
        {
            "Experiment": "Logistic Regression with top 10 ANOVA features",
            "Number_of_features": len(selected_features),
            "Accuracy": accuracy_score(y_test, sel_pred),
            "Precision_malignant": precision_score(y_test, sel_pred, zero_division=0),
            "Recall_malignant": recall_score(y_test, sel_pred, zero_division=0),
            "F1_malignant": f1_score(y_test, sel_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, sel_prob)
        }
    ])
    ablation.to_csv(OUT / "feature_selection_comparison.csv", index=False)
    pd.Series(selected_features, name="Selected feature").to_csv(
        OUT / "selected_features.csv", index=False
    )
    print("\nFEATURE-SELECTION EXPERIMENT")
    print(ablation.round(4).to_string(index=False))
    print("\nTop 10 features selected on training data:")
    print(", ".join(selected_features))

    # Best model's confusion matrix and ROC curve.
    best_name = results.iloc[0]["Model"]
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)
    best_prob = best_model.predict_proba(X_test)[:, 1]
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, best_pred, labels=[0, 1]),
        display_labels=["Benign", "Malignant"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("ROC Curves: Model Comparison")
    plt.tight_layout()
    plt.savefig(OUT / "roc_curves.png", dpi=160)
    plt.close()

    # Model-agnostic, syllabus-aligned feature analysis using Extra Trees importance.
    tree_model = fitted["Extra Trees"]
    importance = pd.Series(
        tree_model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    importance.head(12).sort_values().plot(kind="barh", figsize=(9, 6))
    plt.title("Top 12 Feature Importances (Extra Trees)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=160)
    plt.close()
    importance.rename("importance").to_csv(OUT / "feature_importance.csv")

    with open(OUT / "best_model_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model by test ROC-AUC: {best_name}\n\n")
        f.write(classification_report(
            y_test, best_pred, target_names=["Benign", "Malignant"], zero_division=0
        ))
        f.write("\nNote: this is an educational benchmark, not a clinical diagnostic tool.\n")

    print(f"\nBest model by test ROC-AUC: {best_name}")
    print("\nClassification report:")
    print(classification_report(
        y_test, best_pred, target_names=["Benign", "Malignant"], zero_division=0
    ))
    print(f"\nSaved tables and plots in: {OUT}")
    print("IMPORTANT: These results are for coursework only, not medical diagnosis.")

if __name__ == "__main__":
    main()
