"""Misconception detector for common data science misunderstandings."""
from typing import List, Dict


MISCONCEPTIONS: List[Dict[str, object]] = [
    {
        "id": "accuracy_imbalance",
        "trigger_patterns": ["accuracy", "imbalanced", "99%"],
        "misconception": "High accuracy on imbalanced data means strong performance.",
        "correction": "Accuracy can be misleading on imbalanced datasets. Prefer precision, recall, F1, PR-AUC, or ROC-AUC.",
        "stage": 7,
    },
    {"id": "correlation_causation", "trigger_patterns": ["correlation", "causes"], "misconception": "Correlation implies causation.", "correction": "Correlation measures association, not causal direction.", "stage": 3},
    {"id": "test_tuning", "trigger_patterns": ["tune", "test set"], "misconception": "Use test set to tune hyperparameters.", "correction": "Tune on validation/CV only and keep test set untouched for final evaluation.", "stage": 7},
    {"id": "fit_before_split", "trigger_patterns": ["scale", "before split"], "misconception": "Preprocess full dataset before train-test split.", "correction": "Split first, then fit transforms on train only to prevent leakage.", "stage": 4},
    {"id": "dropna_always", "trigger_patterns": ["dropna", "always"], "misconception": "Dropping all missing rows is always safe.", "correction": "Assess missingness patterns; imputation can preserve data and reduce bias.", "stage": 4},
    {"id": "one_metric_all", "trigger_patterns": ["best metric", "always accuracy"], "misconception": "One metric works for all tasks.", "correction": "Choose metrics by task objective and class balance.", "stage": 7},
    {"id": "more_features_better", "trigger_patterns": ["more features", "always better"], "misconception": "Adding more features always improves performance.", "correction": "Irrelevant features add noise and overfitting risk.", "stage": 5},
    {"id": "pca_for_importance", "trigger_patterns": ["pca", "feature importance"], "misconception": "PCA gives feature importance.", "correction": "PCA components are linear combinations; they are not direct importance scores.", "stage": 5},
    {"id": "no_baseline_needed", "trigger_patterns": ["skip baseline", "start with xgboost"], "misconception": "Baseline models are unnecessary.", "correction": "Start with a baseline to quantify improvements.", "stage": 1},
    {"id": "eda_optional", "trigger_patterns": ["skip eda", "direct modeling"], "misconception": "EDA can be skipped safely.", "correction": "EDA reveals leakage, outliers, and data quality issues before modeling.", "stage": 3},
    {"id": "stratify_not_needed", "trigger_patterns": ["train_test_split", "no stratify"], "misconception": "Stratification is unnecessary for classification.", "correction": "Use stratification to preserve class ratio across splits.", "stage": 2},
    {"id": "shuffle_time_series", "trigger_patterns": ["time series", "shuffle"], "misconception": "Shuffling is always valid.", "correction": "Time series should respect temporal order.", "stage": 2},
    {"id": "scaling_for_trees_required", "trigger_patterns": ["random forest", "must scale"], "misconception": "Tree-based models require scaling.", "correction": "Scaling is usually unnecessary for tree models.", "stage": 6},
    {"id": "normality_required_all", "trigger_patterns": ["must be normal", "all models"], "misconception": "All ML models require normal features.", "correction": "Normality assumptions apply to specific statistical models, not all ML algorithms.", "stage": 4},
    {"id": "high_r2_generalization", "trigger_patterns": ["high r2", "model is good"], "misconception": "High training R2 guarantees generalization.", "correction": "Check validation/test performance to assess generalization.", "stage": 7},
    {"id": "cv_and_test_same", "trigger_patterns": ["cross validation", "test score same"], "misconception": "CV score and test score are interchangeable.", "correction": "CV guides model selection; test score is final unbiased estimate.", "stage": 7},
    {"id": "encoding_target_leakage", "trigger_patterns": ["target encoding", "whole dataset"], "misconception": "Target encoding on full data is fine.", "correction": "Fit target encoding within training folds only.", "stage": 4},
    {"id": "outlier_delete_all", "trigger_patterns": ["outlier", "remove all"], "misconception": "All outliers should be dropped.", "correction": "Validate whether outliers are errors or meaningful rare events.", "stage": 3},
    {"id": "imbalance_smote_test", "trigger_patterns": ["smote", "before split"], "misconception": "Apply SMOTE before splitting data.", "correction": "Apply SMOTE only on training data after split.", "stage": 6},
    {"id": "leakage_feature_creation", "trigger_patterns": ["future data", "feature"], "misconception": "Using future information in features is acceptable.", "correction": "Prevent target and temporal leakage in feature engineering.", "stage": 5},
    {"id": "nulls_meaningless", "trigger_patterns": ["missing values", "ignore"], "misconception": "Missing values are random noise.", "correction": "Missingness can be informative and should be analyzed.", "stage": 3},
    {"id": "precision_recall_confusion", "trigger_patterns": ["precision same recall"], "misconception": "Precision and recall are the same.", "correction": "Precision penalizes false positives; recall penalizes false negatives.", "stage": 7},
    {"id": "xgboost_default_optimal", "trigger_patterns": ["xgboost default", "best"], "misconception": "Default hyperparameters are always close to optimal.", "correction": "Tune key hyperparameters with validation.", "stage": 6},
    {"id": "all_categorical_onehot", "trigger_patterns": ["one hot everything"], "misconception": "One-hot is always best for categorical features.", "correction": "Cardinality and model type should guide encoding choice.", "stage": 4},
    {"id": "data_size_ignore_split", "trigger_patterns": ["small data", "large test"], "misconception": "Any split ratio works regardless of dataset size.", "correction": "Split ratio should reflect sample size and variance needs.", "stage": 2},
    {"id": "auc_for_regression", "trigger_patterns": ["auc", "regression"], "misconception": "AUC is suitable for regression.", "correction": "Use RMSE/MAE/R2 for regression tasks.", "stage": 7},
    {"id": "feature_selection_after_test", "trigger_patterns": ["feature selection", "using test"], "misconception": "Feature selection can use test data.", "correction": "Perform feature selection on training folds only.", "stage": 5},
    {"id": "random_seed_unnecessary", "trigger_patterns": ["random_state", "not needed"], "misconception": "Random seed is not important.", "correction": "Set seeds for reproducibility and fair comparisons.", "stage": 1},
    {"id": "pipeline_unnecessary", "trigger_patterns": ["pipeline", "not needed"], "misconception": "Pipelines are optional even with multiple transforms.", "correction": "Pipelines reduce leakage and keep preprocessing consistent.", "stage": 4},
    {"id": "class_weight_all_models", "trigger_patterns": ["class_weight", "all models"], "misconception": "Every model supports class_weight similarly.", "correction": "Support varies by estimator; verify model-specific parameters.", "stage": 6},
]


class MisconceptionDetector:
    """Matches misconception trigger patterns in query/code text."""

    def __init__(self, bank: List[Dict[str, object]] = None, min_hits: int = 1):
        self.bank = bank or MISCONCEPTIONS
        self.min_hits = max(1, int(min_hits))

    def detect(self, query: str, provided_code: str = "", stage: int = 0) -> List[Dict[str, str]]:
        text = f"{query or ''}\n{provided_code or ''}".lower()
        found: List[Dict[str, str]] = []
        for item in self.bank:
            stage_ok = (stage <= 0) or (int(item.get("stage", 0)) in (0, stage))
            if not stage_ok:
                continue
            patterns = [p.lower() for p in item.get("trigger_patterns", [])]
            hits = sum(1 for p in patterns if p and p in text)
            if hits >= self.min_hits:
                found.append(
                    {
                        "id": str(item.get("id", "")),
                        "misconception": str(item.get("misconception", "")),
                        "correction": str(item.get("correction", "")),
                    }
                )
        return found
