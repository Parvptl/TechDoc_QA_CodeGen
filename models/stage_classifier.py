"""Stage classifier: TF-IDF + LinearSVC (fallback: DistilBERT). Predicts DS pipeline stage 1-7."""
import csv, os, json, pickle
from pathlib import Path

STAGE_NAMES = {1:"Problem Understanding",2:"Data Loading",3:"Exploratory Data Analysis",
               4:"Preprocessing",5:"Feature Engineering",6:"Modeling",7:"Evaluation"}
MODEL_DIR   = "models/stage_classifier"
DATA_PATH   = "data/stage_labeled_dataset.csv"
TFIDF_PATH  = "models/tfidf_svm_fallback.pkl"

def load_data(path=DATA_PATH):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["explanation"] + " " + row["code"])
            labels.append(int(row["predicted_stage"]) - 1)
    return texts, labels

# ── TF-IDF + SVM (fast, no GPU needed) ────────────────────────────────────
def train_tfidf_svm(data_path=DATA_PATH, save_path=TFIDF_PATH):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    import numpy as np

    texts, labels = load_data(data_path)
    labels_1idx = [l+1 for l in labels]

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=15000,
                                   sublinear_tf=True, analyzer="word")),
        ("svm",   CalibratedClassifierCV(LinearSVC(max_iter=3000, C=1.0), cv=3))
    ])
    # Cross-validate first
    scores = cross_val_score(clf, texts, labels_1idx, cv=5, scoring="accuracy")
    print(f"[INFO] TF-IDF+SVM CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    clf.fit(texts, labels_1idx)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[INFO] TF-IDF+SVM saved → {save_path}")
    return clf

# ── DistilBERT fine-tuning (GPU optional) ─────────────────────────────────
def train_bert_classifier(data_path=DATA_PATH, output_dir=MODEL_DIR, epochs=3, batch_size=8):
    try:
        import torch
        from transformers import (DistilBertTokenizerFast,
                                  DistilBertForSequenceClassification,
                                  Trainer, TrainingArguments)
        from torch.utils.data import Dataset
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"[ERROR] {e}. Install with: pip install transformers torch")
        return

    texts, labels = load_data(data_path)
    X_tr, X_val, y_tr, y_val = train_test_split(texts, labels, test_size=0.2,
                                                  random_state=42, stratify=labels)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    class DS(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
            self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=7)
    args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs,
                             per_device_train_batch_size=batch_size,
                             evaluation_strategy="epoch", save_strategy="epoch",
                             load_best_model_at_end=True, logging_steps=10, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=DS(X_tr,y_tr), eval_dataset=DS(X_val,y_val))
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/label_map.json","w") as f:
        json.dump({str(k-1): v for k,v in STAGE_NAMES.items()}, f)
    print(f"[INFO] DistilBERT saved → {output_dir}/")

# ── Inference ─────────────────────────────────────────────────────────────
_tfidf_svm   = None
_bert_model  = None
_bert_tok    = None

def predict_stage(query: str, use_bert: bool = True) -> str:
    """
    Classify query → pipeline stage.
    Returns: "Stage N — Name"  e.g. "Stage 4 — Preprocessing"
    """
    global _tfidf_svm, _bert_model, _bert_tok

    # ── Try DistilBERT first ──────────────────────────────────────────
    if use_bert and Path(MODEL_DIR).exists():
        try:
            import torch
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            if _bert_model is None:
                _bert_tok   = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
                _bert_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
                _bert_model.eval()
            inputs = _bert_tok(query, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                logits = _bert_model(**inputs).logits
            idx = int(logits.argmax(dim=-1))
            return f"Stage {idx+1} — {STAGE_NAMES[idx+1]}"
        except Exception as e:
            print(f"[WARN] BERT inference failed ({e}), using TF-IDF+SVM")

    # ── TF-IDF+SVM fallback ───────────────────────────────────────────
    if Path(TFIDF_PATH).exists():
        if _tfidf_svm is None:
            with open(TFIDF_PATH, "rb") as f:
                _tfidf_svm = pickle.load(f)
        stage_num = int(_tfidf_svm.predict([query])[0])
        return f"Stage {stage_num} — {STAGE_NAMES[stage_num]}"

    # ── Keyword ultra-fallback ────────────────────────────────────────
    return _keyword_fallback(query)

def predict_stage_with_confidence(query: str) -> tuple[str, float]:
    """Returns (stage_string, confidence_score)."""
    global _tfidf_svm
    if Path(TFIDF_PATH).exists():
        if _tfidf_svm is None:
            with open(TFIDF_PATH, "rb") as f:
                _tfidf_svm = pickle.load(f)
        probs = _tfidf_svm.predict_proba([query])[0]
        stage_num = int(_tfidf_svm.classes_[probs.argmax()])
        confidence = float(probs.max())
        return f"Stage {stage_num} — {STAGE_NAMES[stage_num]}", confidence
    return predict_stage(query), 0.9

def _keyword_fallback(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["problem","goal","objective","target","baseline"]): return "Stage 1 — Problem Understanding"
    if any(k in q for k in ["load","read_csv","import","csv","parquet"]):       return "Stage 2 — Data Loading"
    if any(k in q for k in ["plot","visual","distribut","corr","eda","heatmap"]): return "Stage 3 — Exploratory Data Analysis"
    if any(k in q for k in ["missing","null","clean","preprocess","fillna"]):   return "Stage 4 — Preprocessing"
    if any(k in q for k in ["feature","encode","scale","engineer","pca"]):      return "Stage 5 — Feature Engineering"
    if any(k in q for k in ["train","model","fit","classifier","xgboost"]):     return "Stage 6 — Modeling"
    if any(k in q for k in ["evaluat","metric","accuracy","score","auc"]):      return "Stage 7 — Evaluation"
    return "Stage 1 — Problem Understanding"

def extract_stage_num(stage_str: str) -> int:
    """Parse 'Stage N — Name' → N"""
    try:
        return int(stage_str.split("—")[0].replace("Stage","").strip())
    except Exception:
        return 1

if __name__ == "__main__":
    print("\n=== Training TF-IDF+SVM Classifier ===")
    train_tfidf_svm()

    print("\n=== Stage Classifier Demo ===")
    test_queries = [
        "What is the goal of this competition?",
        "How do I load a CSV file with pandas?",
        "Plot the distribution of ages in the dataset",
        "How do I handle missing values in Age column?",
        "How should I encode categorical variables?",
        "How do I train a Random Forest model?",
        "What is the AUC score for my classifier?",
    ]
    for q in test_queries:
        result, conf = predict_stage_with_confidence(q)
        print(f"  Q: {q!r}\n  A: {result} (conf={conf:.3f})\n")
