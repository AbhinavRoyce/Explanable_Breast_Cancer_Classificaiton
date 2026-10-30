# Explainable Breast Cancer Classification Using Machine Learning

## Project summary
This project compares seven classical ML classifiers on the Wisconsin Diagnostic Breast Cancer dataset and tests whether feature selection can reduce the number of features while retaining useful predictive performance.

**Positive class:** malignant (1). **Negative class:** benign (0).

## Why this fits the syllabus
- Supervised learning and classification
- Data inspection and preprocessing
- Feature scaling and feature selection
- Logistic regression, SVM, decision trees, Random Forest, Extra Trees, KNN and Naive Bayes
- Train/test split, stratified 5-fold cross-validation, loss of generalization and model comparison
- Confusion matrix, precision, recall, F1-score, ROC-AUC
- Feature importance and dimensionality reduction/feature selection concepts

## Dataset
The included `data/breast_cancer.csv` contains 569 samples, 30 numeric features and a target column. It is the Wisconsin Diagnostic Breast Cancer dataset distributed by scikit-learn, derived from the UCI dataset.
- 0 = benign
- 1 = malignant
- Dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- Original UCI repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Cite the original dataset in your report.

## Run on macOS / Windows / Linux
Python 3.10 or newer recommended.

```bash
python -m venv .venv
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

Install packages and run:
```bash
pip install -r requirements.txt
python main.py
```

## Outputs created in `outputs/`
- `model_comparison.csv`: test metrics and cross-validation scores for all models
- `feature_selection_comparison.csv`: all 30 features vs top 10 selected features
- `selected_features.csv`: selected feature names
- `confusion_matrix.png`
- `roc_curves.png`
- `feature_importance.png`
- `feature_importance.csv`
- `best_model_report.txt`

## Methodology
1. Inspect the dataset, class balance, missing values and duplicates.
2. Split into 80% training and 20% test data using stratification and a fixed random seed.
3. Scale features for Logistic Regression, SVM and KNN using pipelines. This prevents the scaler from learning from the test data.
4. Train seven baseline classifiers.
5. Evaluate once on the held-out test set and use stratified 5-fold CV on training data.
6. Run a feature-selection experiment: select the top 10 ANOVA features on training data only, then compare against Logistic Regression using all 30 features.
7. Save metrics, plots and a classification report.

## Main contribution
The contribution is a reproducible, fair comparison of seven syllabus-relevant models, plus a feature-selection experiment that tests whether 10 selected features can retain competitive performance compared with all 30 features. Do not claim feature selection improves performance unless your generated results show it.

## Results
Run the program on your machine and use the values in `outputs/model_comparison.csv` and `outputs/feature_selection_comparison.csv`. Results may vary slightly across library versions. Do not invent or copy benchmark numbers without checking your run.

## Limitations and responsible use
This is a small, curated benchmark dataset and results do not establish clinical safety or real-world diagnostic performance. It is for academic learning only, not for medical decisions. A held-out split and cross-validation improve evaluation discipline but do not replace external validation.

## Suggested viva explanation
**Problem:** Can classical ML models distinguish benign from malignant cases using numeric measurements of cell nuclei?

**Baseline:** Logistic Regression, SVM, Decision Tree, Random Forest, KNN, Gaussian Naive Bayes and Extra Trees.

**Evaluation:** Accuracy alone can hide missed malignant cases, so also report malignant-class recall, precision, F1-score and ROC-AUC.

**Contribution:** Compare several models under the same split and test a compact 10-feature Logistic Regression model against the 30-feature version.

**Future work:** nested cross-validation, external validation, calibration, threshold selection based on clinical costs, and explainability with SHAP. These are future work, not claims made by this implementation.
