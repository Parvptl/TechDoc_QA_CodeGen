"""Domain-aware query expansion for DS retrieval."""
from typing import List, Dict


DS_SYNONYMS: Dict[str, List[str]] = {
    "clean data": ["preprocessing", "data wrangling", "handle missing values", "outlier treatment"],
    "missing values": ["imputation", "null handling", "na values", "fillna"],
    "outliers": ["anomaly", "extreme values", "iqr", "z-score"],
    "encode categorical": ["one hot encoding", "label encoding", "ordinal encoding"],
    "scale features": ["normalization", "standardization", "standardscaler", "minmaxscaler"],
    "split data": ["train test split", "validation split", "holdout"],
    "cross validation": ["k-fold", "stratified k-fold", "cv"],
    "class imbalance": ["imbalanced data", "smote", "class weights", "resampling"],
    "feature importance": ["feature selection", "variable importance", "shap"],
    "feature engineering": ["derived features", "feature creation", "transformations"],
    "dimensionality reduction": ["pca", "tsne", "umap"],
    "model performance": ["evaluation metric", "accuracy", "precision", "recall", "f1"],
    "roc": ["auc", "roc-auc", "receiver operating characteristic"],
    "confusion matrix": ["tp fp fn tn", "classification report"],
    "overfitting": ["high variance", "generalization gap", "regularization"],
    "underfitting": ["high bias", "model too simple"],
    "random forest": ["ensemble trees", "bagging", "decision trees"],
    "xgboost": ["gradient boosting", "boosted trees", "gbm"],
    "logistic regression": ["linear classifier", "sigmoid"],
    "linear regression": ["ols", "least squares"],
    "correlation": ["pearson", "spearman", "association"],
    "distribution": ["histogram", "density plot", "kde"],
    "time series": ["temporal data", "date index", "lag features"],
    "text data": ["nlp", "tfidf", "tokenization"],
    "image classification": ["cnn", "computer vision"],
    "pipeline": ["sklearn pipeline", "workflow", "end-to-end"],
    "leakage": ["data leakage", "target leakage", "look-ahead bias"],
    "hyperparameter tuning": ["grid search", "random search", "bayesian optimization"],
    "explain model": ["interpretability", "shap values", "partial dependence"],
    "baseline": ["benchmark model", "dummy classifier", "initial model"],
    "sql": ["query database", "join", "group by"],
    "pandas": ["dataframe", "series", "read_csv"],
    "numpy": ["ndarray", "vectorized operations"],
    "matplotlib": ["plot", "visualization", "chart"],
    "seaborn": ["statistical visualization", "heatmap", "pairplot"],
    "classification": ["predict classes", "binary classification", "multiclass"],
    "regression": ["predict continuous value", "numerical target"],
    "clustering": ["kmeans", "unsupervised learning", "segment"],
    "train model": ["fit estimator", "model training", "learn parameters"],
    "evaluate model": ["test performance", "validation score", "metrics"],
    "precision": ["positive predictive value", "ppv"],
    "recall": ["sensitivity", "true positive rate"],
    "f1": ["harmonic mean", "f1-score"],
    "auc": ["area under curve", "roc auc"],
    "smote": ["oversampling", "synthetic minority"],
    "debug error": ["traceback", "exception", "fix bug"],
}


class QueryExpander:
    """Expands query using domain synonyms and optional planner keywords."""

    def __init__(self, synonyms: Dict[str, List[str]] = None):
        self.synonyms = synonyms or DS_SYNONYMS

    def expand(self, query: str, extra_terms: List[str] = None) -> str:
        q = (query or "").strip()
        q_lower = q.lower()
        terms = []
        for key, values in self.synonyms.items():
            if key in q_lower:
                terms.extend(values)
        if extra_terms:
            terms.extend([t for t in extra_terms if t])

        dedup = []
        seen = set()
        for t in terms:
            token = t.strip().lower()
            if token and token not in seen:
                seen.add(token)
                dedup.append(t.strip())
        if not dedup:
            return q
        return f"{q} {' '.join(dedup)}"
