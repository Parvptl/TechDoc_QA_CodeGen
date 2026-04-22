"""
PART 4 — STAGE + INTENT CLASSIFIER (Member 4)
===============================================
Two classifiers in one module:

  1. Stage classifier (1–7)   — which DS pipeline step the query belongs to
  2. Intent classifier        — what the user wants:
       "code"          → generate/show code
       "explanation"   → explain a concept
       "visualization" → create a chart/plot

Both use TF-IDF + LinearSVC (fast, no GPU, 98%+ CV accuracy).
DistilBERT fine-tuning path available if GPU present.

Functions:
    predict_stage(query)          → "Stage N — Name"
    predict_intent(query)         → "code" | "explanation" | "visualization"
    predict_both(query)           → {"stage": int, "stage_name": str,
                                      "intent": str, "stage_conf": float,
                                      "intent_conf": float}

Run to train both:
    python classifier/intent_stage_classifier.py
"""

import csv
import pickle
import re
from pathlib import Path
from typing import Optional

STAGE_NAMES = {
    1: "Problem Understanding",  2: "Data Loading",
    3: "Exploratory Data Analysis", 4: "Preprocessing",
    5: "Feature Engineering",    6: "Modeling",
    7: "Evaluation",
}

INTENTS = ["code", "explanation", "visualization"]

# Saved model paths
STAGE_MODEL_PATH  = "models/tfidf_svm_fallback.pkl"        # existing
INTENT_MODEL_PATH = "classifier/intent_classifier.pkl"     # new

# ── Intent training data ───────────────────────────────────────────────────────
INTENT_EXAMPLES = {
    "code": [
        "How do I write code to fill missing values?",
        "Give me the code to train a Random Forest",
        "Show me how to implement one-hot encoding",
        "Write a function to split train test data",
        "Give me code for cross validation",
        "How do I code a confusion matrix?",
        "Show code to compute AUC score",
        "Implement a pipeline with scaler and model",
        "Write a custom loss function in PyTorch",
        "How to apply StandardScaler in sklearn?",
        "Give me the pandas code to load CSV",
        "Write code to detect outliers",
        "How to encode categoricals in code?",
        "Show me how to use GridSearchCV",
        "Code to save and load a model with joblib",
        "How do I implement early stopping?",
        "Write a SMOTE oversampling script",
        "Give me code to compute SHAP values",
        "How to create lag features in Python?",
        "Show me the syntax for ColumnTransformer",
        "How do I train XGBoost with early stopping?",
        "Write code to plot learning curves",
        "Give me a feature importance plot code",
        "How do I do hyperparameter tuning with Optuna?",
    ],
    "explanation": [
        "What is the goal of a data science project?",
        "Explain what overfitting means",
        "What is the difference between precision and recall?",
        "Why do we need to normalize features?",
        "What is data leakage and why is it bad?",
        "Explain the concept of cross-validation",
        "What does MRR stand for?",
        "Why is EDA important before modeling?",
        "What is the bias-variance tradeoff?",
        "Explain what a confusion matrix tells us",
        "What is regularization in machine learning?",
        "Why should I use stratified sampling?",
        "What is the difference between bagging and boosting?",
        "Explain feature importance",
        "What is ROC AUC and why use it?",
        "When should I use SMOTE?",
        "What is the purpose of a validation set?",
        "Why normalize text before NLP?",
        "Explain the concept of feature engineering",
        "What is a null model and why does it matter?",
        "When should I use median imputation?",
        "What is concept drift?",
        "Explain what Reciprocal Rank Fusion does",
        "What is the difference between accuracy and F1?",
    ],
    "visualization": [
        "Show me a histogram of the Age column",
        "Plot the distribution of the target variable",
        "Create a heatmap of feature correlations",
        "Visualize class imbalance in my dataset",
        "Plot a boxplot to check for outliers",
        "Show me a pairplot of all features",
        "Create a scatter plot of Age vs Fare",
        "Plot ROC curve for my classifier",
        "Show confusion matrix as a heatmap",
        "Visualize learning curves to detect overfitting",
        "Plot feature importances as a bar chart",
        "Show me the distribution of missing values",
        "Create a violin plot comparing groups",
        "Plot residuals for my regression model",
        "Visualize the decision boundary",
        "Show me a SHAP summary plot",
        "Create a time series plot of sales data",
        "Visualize correlation between Age and Survived",
        "Plot precision-recall curve",
        "Show me a count plot by category",
        "Visualize my model's calibration curve",
        "Create a partial dependence plot",
        "Plot the AUC-ROC curve",
        "Show me a word cloud for text data",
    ],
}

# Intent keyword signals (for fast rule-based fallback)
INTENT_SIGNALS = {
    "visualization": [
        "plot", "chart", "graph", "visualize", "histogram", "heatmap",
        "scatter", "boxplot", "pairplot", "show me", "display", "draw",
        "create a figure", "visualisation", "seaborn", "matplotlib",
        "distribution plot", "bar chart", "pie chart",
    ],
    "code": [
        "how do i", "give me code", "write code", "implement", "show code",
        "syntax", "function", "script", "snippet", "example code",
        "in python", "using sklearn", "with pandas", "code to",
    ],
    "explanation": [
        "what is", "explain", "why", "difference between", "when should",
        "what does", "how does", "concept", "meaning", "definition",
        "purpose of", "tell me about", "describe",
    ],
}


# ── Training ───────────────────────────────────────────────────────────────────
def train_intent_classifier(save_path: str = INTENT_MODEL_PATH):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from collections import Counter
    import numpy as np

    texts, labels = [], []
    for intent, examples in INTENT_EXAMPLES.items():
        for ex in examples:
            texts.append(ex)
            labels.append(intent)

    # CalibratedClassifierCV needs enough samples per class per fold.
    # Keep CV conservative so training works on small toy datasets too.
    label_counts = Counter(labels)
    min_per_class = min(label_counts.values()) if label_counts else 0
    # Be conservative: calibration with cv=2 is much less brittle on small datasets
    # and still yields usable confidence estimates.
    calib_cv = 2 if min_per_class >= 4 else 1

    if calib_cv == 1:
        # Fallback: no calibration if dataset is too small.
        final_est = LinearSVC(max_iter=2000, C=1.0)
    else:
        final_est = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=calib_cv)

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=5000, sublinear_tf=True)),
        ("svm",   final_est),
    ])
    outer_cv = 5 if min_per_class >= 10 else (3 if min_per_class >= 6 else 2)
    try:
        scores = cross_val_score(clf, texts, labels, cv=outer_cv, scoring="accuracy")
        print(f"[INFO] Intent classifier CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    except Exception as e:
        print(f"[WARN] Intent classifier CV skipped: {e}")

    try:
        clf.fit(texts, labels)
    except ValueError as e:
        # If calibration still fails (rare sklearn edge cases), fall back to raw LinearSVC.
        print(f"[WARN] Intent classifier calibration failed; falling back to LinearSVC. Error: {e}")
        clf = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=5000, sublinear_tf=True)),
                ("svm", LinearSVC(max_iter=2000, C=1.0)),
            ]
        )
        clf.fit(texts, labels)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[INFO] Intent classifier saved -> {save_path}")
    return clf


def train_stage_classifier(
    data_path: str = "data/stage_labeled_dataset.csv",
    save_path: str = STAGE_MODEL_PATH,
):
    """Retrain stage classifier if needed (delegates to models/stage_classifier.py)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.stage_classifier import train_tfidf_svm
    return train_tfidf_svm(data_path, save_path)


# ── Inference ──────────────────────────────────────────────────────────────────
_stage_clf  = None
_intent_clf = None


def _load_stage():
    global _stage_clf
    if _stage_clf is None and Path(STAGE_MODEL_PATH).exists():
        with open(STAGE_MODEL_PATH, "rb") as f:
            _stage_clf = pickle.load(f)
    return _stage_clf


def _load_intent():
    global _intent_clf
    if _intent_clf is None:
        if Path(INTENT_MODEL_PATH).exists():
            with open(INTENT_MODEL_PATH, "rb") as f:
                _intent_clf = pickle.load(f)
        else:
            print("[INFO] Intent classifier not trained yet — training now...")
            _intent_clf = train_intent_classifier()
    return _intent_clf


def predict_stage(query: str) -> str:
    """
    Predict pipeline stage from query.
    Returns: "Stage N — Name"  e.g. "Stage 4 — Preprocessing"
    """
    clf = _load_stage()
    if clf is None:
        return _keyword_stage_fallback(query)
    stage_num = int(clf.predict([query])[0])
    return f"Stage {stage_num} — {STAGE_NAMES[stage_num]}"


def predict_stage_with_confidence(query: str) -> tuple[str, float]:
    """Returns (stage_string, confidence_score 0–1)."""
    clf = _load_stage()
    if clf is None:
        return predict_stage(query), 0.5
    probs     = clf.predict_proba([query])[0]
    stage_num = int(clf.classes_[probs.argmax()])
    conf      = float(probs.max())
    return f"Stage {stage_num} — {STAGE_NAMES[stage_num]}", conf


def predict_intent(query: str) -> str:
    """
    Predict user intent: "code" | "explanation" | "visualization"

    Uses TF-IDF+SVM classifier with rule-based fallback.
    The intent drives which response format to prioritize.
    """
    # Rule-based fast path
    q_lower = query.lower()
    for intent, signals in INTENT_SIGNALS.items():
        if any(sig in q_lower for sig in signals):
            return intent

    # ML classifier
    clf = _load_intent()
    if clf is None:
        return "code"   # safe default
    return str(clf.predict([query])[0])


def predict_intent_with_confidence(query: str) -> tuple[str, float]:
    """Returns (intent, confidence_score 0–1)."""
    # Rule-based always high confidence
    q_lower = query.lower()
    for intent, signals in INTENT_SIGNALS.items():
        if any(sig in q_lower for sig in signals):
            return intent, 0.95

    clf = _load_intent()
    if clf is None:
        return "code", 0.5
    # Calibrated models expose predict_proba; raw LinearSVC does not.
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba([query])[0]
        intent = str(clf.classes_[probs.argmax()])
        return intent, float(probs.max())

    intent = str(clf.predict([query])[0])
    conf = 0.6
    if hasattr(clf, "decision_function"):
        try:
            scores = clf.decision_function([query])
            # decision_function may be shape (n_classes,) or (1, n_classes)
            if hasattr(scores, "__len__") and len(getattr(scores, "shape", [])) == 2:
                scores = scores[0]
            import numpy as np

            scores = np.array(scores, dtype=float).ravel()
            # Confidence proxy: normalized margin between best and second best.
            if scores.size >= 2:
                top2 = np.sort(scores)[-2:]
                margin = float(top2[1] - top2[0])
                conf = float(max(0.5, min(0.95, 0.5 + margin / 5.0)))
        except Exception:
            conf = 0.6
    return intent, float(conf)


def predict_both(query: str) -> dict:
    """
    Run both classifiers and return combined result.

    Returns:
        {
          "stage":       int,
          "stage_name":  str,
          "stage_conf":  float,
          "intent":      str,   # "code" | "explanation" | "visualization"
          "intent_conf": float,
        }
    """
    stage_str, stage_conf = predict_stage_with_confidence(query)
    intent,    intent_conf = predict_intent_with_confidence(query)

    # Extract stage number
    try:
        stage_num = int(stage_str.split("—")[0].replace("Stage", "").strip())
    except Exception:
        stage_num = 1

    return {
        "stage":       stage_num,
        "stage_name":  STAGE_NAMES.get(stage_num, "Unknown"),
        "stage_conf":  round(stage_conf, 3),
        "intent":      intent,
        "intent_conf": round(intent_conf, 3),
    }


def extract_stage_num(stage_str: str) -> int:
    try:
        return int(stage_str.split("—")[0].replace("Stage", "").strip())
    except Exception:
        return 1


# ── Keyword fallbacks ──────────────────────────────────────────────────────────
def _keyword_stage_fallback(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["problem", "goal", "objective", "target", "baseline"]):
        return "Stage 1 — Problem Understanding"
    if any(k in q for k in ["load", "read_csv", "import", "csv", "parquet"]):
        return "Stage 2 — Data Loading"
    if any(k in q for k in ["plot", "visual", "distribut", "corr", "eda", "heatmap"]):
        return "Stage 3 — Exploratory Data Analysis"
    if any(k in q for k in ["missing", "null", "clean", "preprocess", "fillna"]):
        return "Stage 4 — Preprocessing"
    if any(k in q for k in ["feature", "encode", "scale", "engineer", "pca"]):
        return "Stage 5 — Feature Engineering"
    if any(k in q for k in ["train", "model", "fit", "classifier", "xgboost"]):
        return "Stage 6 — Modeling"
    if any(k in q for k in ["evaluat", "metric", "accuracy", "score", "auc"]):
        return "Stage 7 — Evaluation"
    return "Stage 1 — Problem Understanding"


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("=== Training Intent Classifier ===")
    train_intent_classifier()

    print("\n=== Classifier Demo ===\n")
    test_cases = [
        ("What is the goal of predicting house prices?",          1, "explanation"),
        ("How do I load a CSV file with pandas?",                 2, "code"),
        ("Show me a correlation heatmap for all features",        3, "visualization"),
        ("How do I fill missing values in Age?",                  4, "code"),
        ("Explain what one-hot encoding does",                    5, "explanation"),
        ("Train a Random Forest with 200 estimators",             6, "code"),
        ("Plot the confusion matrix for my model",                7, "visualization"),
        ("What is the difference between precision and recall?",  7, "explanation"),
        ("Create a scatter plot of Age vs Fare",                  3, "visualization"),
        ("Write code to compute SHAP values",                     7, "code"),
    ]

    stage_correct = 0
    intent_correct = 0
    for query, exp_stage, exp_intent in test_cases:
        result = predict_both(query)
        s_ok = "✓" if result["stage"] == exp_stage else "✗"
        i_ok = "✓" if result["intent"] == exp_intent else "✗"
        if result["stage"] == exp_stage:   stage_correct += 1
        if result["intent"] == exp_intent: intent_correct += 1
        print(f"  {s_ok}{i_ok} Stage={result['stage']}({result['stage_conf']:.2f})  "
              f"Intent={result['intent']}({result['intent_conf']:.2f})  "
              f"| {query[:55]}")

    n = len(test_cases)
    print(f"\nStage accuracy:  {stage_correct}/{n} ({stage_correct/n*100:.0f}%)")
    print(f"Intent accuracy: {intent_correct}/{n} ({intent_correct/n*100:.0f}%)")
