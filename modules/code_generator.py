"""
PART 3b — CODE GENERATION MODULE (Member 3)
============================================
Generates context-aware Python code for data science tasks.
Architecture:
  1. Template-based generation (fast, always works)
  2. RAG-augmented generation (retrieves and adapts closest code)
  3. DistilBERT/CodeT5 fine-tuning path (optional, needs GPU)

This module is the primary code generation component —
the RAG retriever returns verbatim code, but this module
ADAPTS it based on user context (column names, model choices, etc.)
"""
import re
from typing import Optional

STAGE_NAMES = {
    1:"Problem Understanding", 2:"Data Loading", 3:"Exploratory Data Analysis",
    4:"Preprocessing", 5:"Feature Engineering", 6:"Modeling", 7:"Evaluation",
}

# ── Context extraction from user query ────────────────────────────────────────
def extract_context(query: str) -> dict:
    """
    Extract contextual hints from user query:
    column names, model types, file paths, metrics, etc.
    """
    ctx = {
        "columns":  [],
        "model":    None,
        "metric":   None,
        "filepath": None,
        "strategy": None,
        "n":        None,
    }
    q = query.lower()

    # Extract quoted column names e.g. 'Age', "Fare"
    ctx["columns"] = re.findall(r"['\"]([A-Za-z_]\w*)['\"]", query)

    # Detect model type
    model_patterns = {
        "random_forest": r"random\s*forest|rf\b",
        "xgboost":       r"xgboost|xgb\b",
        "lightgbm":      r"lightgbm|lgbm\b",
        "logistic":      r"logistic",
        "svm":           r"\bsvm\b|support\s*vector",
        "neural":        r"neural|mlp|deep\s*learn",
        "decision_tree": r"decision\s*tree",
        "catboost":      r"catboost",
    }
    for name, pat in model_patterns.items():
        if re.search(pat, q):
            ctx["model"] = name
            break

    # Detect metric
    metric_patterns = {
        "roc_auc":  r"auc|roc",
        "accuracy": r"accuracy",
        "f1":       r"\bf1\b|f1.score",
        "rmse":     r"rmse|root.mean.square",
        "mae":      r"\bmae\b|mean.absolute",
        "r2":       r"\br2\b|r.squared",
    }
    for name, pat in metric_patterns.items():
        if re.search(pat, q):
            ctx["metric"] = name
            break

    # Detect file path
    path_match = re.search(r"[\w./\\-]+\.(?:csv|xlsx|parquet|json|tsv)", query, re.I)
    if path_match:
        ctx["filepath"] = path_match.group()

    # Detect imputation strategy
    if re.search(r"median", q):  ctx["strategy"] = "median"
    elif re.search(r"mean", q):  ctx["strategy"] = "mean"
    elif re.search(r"mode|most.frequent", q): ctx["strategy"] = "most_frequent"
    elif re.search(r"knn", q):   ctx["strategy"] = "knn"

    # Detect N (number of estimators, folds, etc.)
    n_match = re.search(r"\b(\d+)\s*(?:estimators|trees|folds|neighbors|components|features)\b", q)
    if n_match:
        ctx["n"] = int(n_match.group(1))

    return ctx


# ── Code templates per stage ──────────────────────────────────────────────────
CODE_TEMPLATES = {

1: lambda ctx: (
    "# Problem definition\n"
    f"task_type   = 'Binary Classification'\n"
    f"target_col  = {repr(ctx['columns'][0] if ctx['columns'] else 'target')}\n"
    f"eval_metric = {repr(ctx['metric'] or 'roc_auc')}\n\n"
    "print(f'Task:   {task_type}')\n"
    "print(f'Target: {target_col}')\n"
    "print(f'Metric: {eval_metric}')\n\n"
    "# Null baseline — your model must beat this\n"
    "from sklearn.dummy import DummyClassifier\n"
    "from sklearn.model_selection import cross_val_score\n"
    "baseline = DummyClassifier(strategy='most_frequent')\n"
    "scores = cross_val_score(baseline, X, y, cv=5, scoring=eval_metric)\n"
    "print(f'Baseline {eval_metric}: {scores.mean():.4f}')"
),

2: lambda ctx: f'''import pandas as pd
filepath = {repr(ctx["filepath"] or "data/train.csv")}
df = pd.read_csv(filepath)
print(f"Shape: {{df.shape}}")
print(f"Columns: {{df.columns.tolist()}}")
print(f"\\nDtypes:\\n{{df.dtypes}}")
print(f"\\nMissing values:\\n{{df.isnull().sum()[df.isnull().sum()>0]}}")
print(f"\\nSample:\\n{{df.head(3)}}")''',

3: lambda ctx: f'''import matplotlib.pyplot as plt
import seaborn as sns

col = {repr(ctx["columns"][0]) if ctx["columns"] else repr("feature")}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distribution
sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[0])
axes[0].set_title(f"Distribution of {{col}}")

# Box plot
sns.boxplot(y=df[col].dropna(), ax=axes[1])
axes[1].set_title(f"Boxplot — {{col}}")

# vs target
if "target" in df.columns:
    sns.boxplot(data=df, x="target", y=col, ax=axes[2])
    axes[2].set_title(f"{{col}} by Target")

plt.tight_layout(); plt.show()

# Stats
print(f"{{col}} stats:\\n{{df[col].describe()}}")
print(f"Skewness: {{df[col].skew():.3f}}")
print(f"Missing: {{df[col].isnull().sum()}} ({{df[col].isnull().mean()*100:.1f}}%)")''',

4: lambda ctx: _preprocessing_code(ctx),

5: lambda ctx: f'''from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

col = {repr(ctx["columns"][0]) if ctx["columns"] else repr("feature")}

if df[col].dtype == "object":
    # Categorical: encode
    if df[col].nunique() <= 5:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        print(f"One-hot encoded {{col}}: {{dummies.columns.tolist()}}")
    else:
        le = LabelEncoder()
        df[f"{{col}}_enc"] = le.fit_transform(df[col].fillna("Unknown"))
        print(f"Label encoded {{col}}: {{dict(zip(le.classes_, le.transform(le.classes_)))}}")
else:
    # Numeric: scale
    import numpy as np
    df[f"{{col}}_log"] = np.log1p(df[col])
    print(f"Log-transformed {{col}}: skew {{df[col].skew():.2f}} → {{df[f'{{col}}_log'].skew():.2f}}")
    scaler = StandardScaler()
    df[f"{{col}}_scaled"] = scaler.fit_transform(df[[col]])
    print(f"Standardized {{col}}: mean={{df[f'{{col}}_scaled'].mean():.3f}}")''',

6: lambda ctx: _modeling_code(ctx),

7: lambda ctx: _evaluation_code(ctx),
}

def _preprocessing_code(ctx):
    col = repr(ctx["columns"][0]) if ctx["columns"] else repr("feature")
    strategy = ctx.get("strategy") or "median"
    if strategy == "knn":
        return f'''from sklearn.impute import KNNImputer
import pandas as pd

num_cols = df.select_dtypes("number").columns.tolist()
print(f"Missing before: {{df[num_cols].isnull().sum().sum()}}")
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])
print(f"Missing after KNN imputation: {{df[num_cols].isnull().sum().sum()}}")'''
    else:
        return f'''from sklearn.impute import SimpleImputer
import pandas as pd

col = {col}
strategy = {repr(strategy)}

# Single column imputation
imputer = SimpleImputer(strategy=strategy)
df[col] = imputer.fit_transform(df[[col]])
print(f"Nulls in {{col}} after {{strategy}} imputation: {{df[col].isnull().sum()}}")

# Full numeric imputation
num_cols = df.select_dtypes("number").columns
df[num_cols] = SimpleImputer(strategy=strategy).fit_transform(df[num_cols])

# Categorical: mode
cat_cols = df.select_dtypes("object").columns
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

print(f"Total remaining nulls: {{df.isnull().sum().sum()}}")'''

def _modeling_code(ctx):
    model_type = ctx.get("model") or "random_forest"
    n = ctx.get("n") or 100
    metric = ctx.get("metric") or "roc_auc"

    model_snippets = {
        "random_forest": f"from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators={n}, max_depth=8, random_state=42)",
        "xgboost":       f"from xgboost import XGBClassifier\nmodel = XGBClassifier(n_estimators={n}, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='logloss', random_state=42)",
        "lightgbm":      f"import lightgbm as lgb\nmodel = lgb.LGBMClassifier(n_estimators={n}, learning_rate=0.05, num_leaves=31, random_state=42)",
        "logistic":      f"from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(C=1.0, max_iter=1000, random_state=42)",
        "svm":           f"from sklearn.svm import SVC\nmodel = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)",
        "neural":        f"from sklearn.neural_network import MLPClassifier\nmodel = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=300, random_state=42)",
        "decision_tree": f"from sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier(max_depth=5, random_state=42)",
    }
    model_init = model_snippets.get(model_type, model_snippets["random_forest"])

    return f'''from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split

{model_init}

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring={repr(metric)})
print(f"CV {metric}: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")

# Final fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
print(f"Train {metric}: {{cross_val_score(model, X_train, y_train, cv=3, scoring={repr(metric)}).mean():.4f}}")'''

def _evaluation_code(ctx):
    metric = ctx.get("metric") or "roc_auc"
    return f'''from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    ConfusionMatrixDisplay, roc_curve
)
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Core metrics
auc  = roc_auc_score(y_test, y_prob)
f1   = f1_score(y_test, y_pred, average="weighted")
print(f"AUC:  {{auc:.4f}}")
print(f"F1:   {{f1:.4f}}")
print(f"\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, label=f"AUC={{auc:.4f}}")
axes[0].plot([0,1],[0,1],"k--")
axes[0].set_title("ROC Curve"); axes[0].legend()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[1], cmap="Blues")
axes[1].set_title("Confusion Matrix")

plt.tight_layout(); plt.show()'''


# ── Main entry point ──────────────────────────────────────────────────────────
def generate_code(query: str, stage_num: int,
                  retrieved_code: Optional[str] = None) -> dict:
    """
    Generate context-aware code for a query.

    Strategy:
      1. Extract context from query (column names, model type, etc.)
      2. If retrieved_code exists and is very relevant → adapt it
      3. Otherwise → use template for the stage

    Returns:
        {
          "code":    str,   # generated Python code
          "method":  str,   # "template" | "adapted" | "retrieved"
          "context": dict,  # extracted context
        }
    """
    ctx = extract_context(query)

    # If retrieved code is short and clean, adapt it
    if retrieved_code and len(retrieved_code) < 600:
        adapted = _adapt_retrieved_code(retrieved_code, ctx)
        if adapted:
            return {"code": adapted, "method": "adapted", "context": ctx}

    # Use template
    template_fn = CODE_TEMPLATES.get(stage_num)
    if template_fn:
        try:
            code = template_fn(ctx)
            return {"code": code, "method": "template", "context": ctx}
        except Exception as e:
            pass

    # Fallback: return retrieved as-is
    if retrieved_code:
        return {"code": retrieved_code, "method": "retrieved", "context": ctx}

    return {"code": "# No code generated for this query", "method": "none", "context": ctx}


def _adapt_retrieved_code(code: str, ctx: dict) -> Optional[str]:
    """Adapt retrieved code by substituting context-specific values."""
    adapted = code
    # Substitute column names
    if ctx["columns"]:
        col = ctx["columns"][0]
        # Replace generic column references
        adapted = re.sub(r'"Age"', f'"{col}"', adapted)
        adapted = re.sub(r"'Age'", f"'{col}'", adapted)

    # Substitute file path
    if ctx["filepath"]:
        adapted = re.sub(r'"train\.csv"|\'train\.csv\'',
                         repr(ctx["filepath"]), adapted)

    # Substitute n_estimators
    if ctx["n"]:
        adapted = re.sub(r"n_estimators=\d+", f"n_estimators={ctx['n']}", adapted)

    # Only return adapted if something actually changed
    return adapted if adapted != code else None


def validate_code_syntax(code: str) -> dict:
    """Check if code is syntactically valid Python."""
    import ast
    try:
        ast.parse(code)
        return {"valid": True, "error": None}
    except SyntaxError as e:
        return {"valid": False, "error": str(e)}


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("What is the goal of predicting 'Survived' using AUC?", 1),
        ("How do I load 'data/titanic.csv' with pandas?", 2),
        ("Show distribution of 'Age' column", 3),
        ("Fill missing values in 'Fare' using median", 4),
        ("Encode the 'Sex' column", 5),
        ("Train a Random Forest with 200 estimators", 6),
        ("Compute AUC and confusion matrix", 7),
    ]
    print("=== Code Generator Demo ===\n")
    for query, stage in test_cases:
        result = generate_code(query, stage)
        validation = validate_code_syntax(result["code"])
        status = "✓" if validation["valid"] else f"✗ {validation['error']}"
        print(f"Q: {query}")
        print(f"Stage {stage} | Method: {result['method']} | Syntax: {status}")
        print(f"Context: {result['context']}")
        print(f"Code preview: {result['code'][:100]}...")
        print()
