"""Applies keyword rules + TF-IDF ensemble to label dataset.csv rows with DS pipeline stages."""
import csv, re
from pathlib import Path

STAGE_NAMES = {
    1:"Problem Understanding", 2:"Data Loading", 3:"Exploratory Data Analysis",
    4:"Preprocessing", 5:"Feature Engineering", 6:"Modeling", 7:"Evaluation",
}
STAGE_KEYWORDS = {
    1:["problem","objective","goal","business","define","task","competition","target variable",
       "predict","baseline","kpi","scope","hypothesis","success metric","null model"],
    2:["read_csv","load_data","pd.read","import pandas","import numpy","load dataset",
       "read data","data loading","open file","glob","json.load","read_excel","read_sql",
       "database","api","fetch","parquet","encoding","chunk"],
    3:["eda","exploratory","distribution","histogram","boxplot","countplot","correlation",
       "heatmap","pairplot","describe","value_counts","shape","info()","visualize","plot",
       "seaborn","matplotlib","skewness","kurtosis","missing pattern","nunique"],
    4:["missing","null","fillna","dropna","impute","clean","outlier","duplicate","strip",
       "replace","dtype","astype","preprocessing","handle nan","isnull","winsorize","clip",
       "smote","imbalance","normalize text","regex"],
    5:["feature","engineer","polynomial","interaction","encode","labelencoder","onehotencoder",
       "get_dummies","scaling","standardscaler","minmaxscaler","pca","tfidf","embedding",
       "new column","derive","bin","log transform","target encode","frequency encode"],
    6:["model","train","fit","predict","random forest","xgboost","lgbm","logistic regression",
       "svm","neural network","sklearn","cross_val","gridsearch","pipeline","classifier",
       "regressor","hyperparameter","optuna","stacking","ensemble"],
    7:["accuracy","f1_score","roc_auc","precision","recall","confusion matrix","classification_report",
       "mse","rmse","mae","evaluate","metric","score","performance","test set","learning curve",
       "overfitting","underfitting","shap","feature importance","calibration"],
}

def rule_based_label(text):
    combined = text.lower()
    scores = {s: sum(1 for kw in kws if kw in combined) for s, kws in STAGE_KEYWORDS.items()}
    total = sum(scores.values())
    best = max(scores, key=scores.get)
    conf = round(scores[best] / total, 3) if total > 0 else 0.0
    return best, conf

def train_tfidf_clf(rows):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        texts = [r["explanation"] + " " + r["code"] for r in rows]
        labels = [int(r["pipeline_stage"]) for r in rows]
        clf = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)),
                        ("lr", LogisticRegression(max_iter=500, C=1.0, random_state=42))])
        clf.fit(texts, labels)
        print("[INFO] TF-IDF classifier trained.")
        return clf
    except Exception as e:
        print(f"[WARN] Classifier training failed: {e}")
        return None

def label_dataset(inp="data/dataset.csv", out="data/stage_labeled_dataset.csv"):
    if not Path(inp).exists():
        raise FileNotFoundError(f"{inp} not found. Run create_dataset.py first.")
    with open(inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[INFO] Loaded {len(rows)} rows from {inp}")
    clf = train_tfidf_clf(rows)
    labeled = []
    for row in rows:
        combined = row["explanation"] + " " + row["code"]
        rule_stage, rule_conf = rule_based_label(combined)
        if clf:
            probs = clf.predict_proba([combined])[0]
            ml_stage = int(clf.classes_[probs.argmax()])
            ml_conf = round(float(probs.max()), 3)
            if ml_conf >= rule_conf:
                final_stage, final_conf, method = ml_stage, ml_conf, "ml"
            else:
                final_stage, final_conf, method = rule_stage, rule_conf, "rule"
        else:
            final_stage, final_conf, method = rule_stage, rule_conf, "rule"
        labeled.append({
            "explanation": row["explanation"], "code": row["code"],
            "original_stage": row["pipeline_stage"],
            "predicted_stage": final_stage,
            "stage_name": STAGE_NAMES[final_stage],
            "confidence": final_conf, "label_method": method,
            "source": row.get("source","curated"),
            "difficulty": row.get("difficulty","intermediate"),
        })
    agree = sum(1 for r in labeled if str(r["original_stage"])==str(r["predicted_stage"]))
    print(f"[INFO] Label agreement: {agree}/{len(labeled)} ({100*agree/len(labeled):.1f}%)")
    fieldnames = ["explanation","code","original_stage","predicted_stage","stage_name","confidence","label_method","source","difficulty"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        csv.DictWriter(f, fieldnames=fieldnames).writerows(labeled)
    print(f"[INFO] Saved → {out}")
    return labeled

if __name__ == "__main__":
    label_dataset()
