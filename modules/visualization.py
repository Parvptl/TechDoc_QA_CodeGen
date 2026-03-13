"""
PART 4 — VISUALIZATION EXECUTION MODULE
Routes EDA queries (Stage 3) to safe code generation + execution.
Returns matplotlib figure as base64 PNG or file path.
Depends on: stage_classifier.predict_stage()
"""

import io
import re
import base64
import traceback
import textwrap
from typing import Optional

# ── Safe built-in allowlist ──────────────────────────────────────────────────
SAFE_IMPORTS = {
    "pandas", "numpy", "matplotlib", "matplotlib.pyplot",
    "seaborn", "sklearn", "scipy", "math",
}

BANNED_TOKENS = [
    "os.system", "subprocess", "eval(", "exec(", "__import__",
    "open(", "shutil", "socket", "requests", "urllib",
]

# ── EDA query → code templates ───────────────────────────────────────────────
EDA_TEMPLATES = {
    r"distribut|histogram": """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

np.random.seed(42)
data = np.random.normal(50, 15, 300)
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data, bins=30, kde=True, color='steelblue', ax=ax)
ax.set_title('Feature Distribution', fontsize=14)
ax.set_xlabel('Value'); ax.set_ylabel('Frequency')
plt.tight_layout()
""",
    r"correlation|heatmap": """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(42)
df = pd.DataFrame(np.random.randn(100, 5),
                  columns=['Age', 'Fare', 'Pclass', 'SibSp', 'Parch'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
""",
    r"boxplot|outlier|box plot": """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(42)
df = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], 150),
    'Value':    np.random.normal(50, 20, 150)
})
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='Category', y='Value', palette='Set2', ax=ax)
ax.set_title('Boxplot by Category', fontsize=14)
plt.tight_layout()
""",
    r"count|bar.*plot|class.*imbalanc": """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(42)
counts = pd.Series({'Not Survived': 549, 'Survived': 342})
fig, ax = plt.subplots(figsize=(7, 5))
counts.plot(kind='bar', color=['#e74c3c', '#2ecc71'], ax=ax)
ax.set_title('Target Variable Distribution', fontsize=14)
ax.set_ylabel('Count'); ax.set_xlabel('')
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12)
plt.xticks(rotation=0); plt.tight_layout()
""",
    r"scatter|relation|trend": """
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.rand(100) * 100
y = 2.5 * x + np.random.randn(100) * 15
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, alpha=0.6, color='steelblue', edgecolors='white')
m, b = np.polyfit(x, y, 1)
ax.plot(x, m*x + b, 'r--', label=f'Trend y={m:.2f}x+{b:.1f}')
ax.set_xlabel('Feature X'); ax.set_ylabel('Target Y')
ax.set_title('Scatter Plot with Trend Line', fontsize=14)
ax.legend(); plt.tight_layout()
""",
    r"pairplot|pair.*plot|feature.*relation": """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(42)
df = pd.DataFrame({
    'Age':      np.random.randint(18, 80, 100),
    'Fare':     np.random.exponential(30, 100),
    'Pclass':   np.random.choice([1, 2, 3], 100),
    'Survived': np.random.choice([0, 1], 100),
})
fig = sns.pairplot(df, hue='Survived', height=2.2,
                   palette={0: '#e74c3c', 1: '#2ecc71'}).fig
fig.suptitle('Pairplot — Feature Relationships', y=1.02, fontsize=13)
plt.tight_layout()
""",
}

DEFAULT_CODE = EDA_TEMPLATES[r"distribut|histogram"]  # safe default


# ── Safety checker ────────────────────────────────────────────────────────────
def is_safe_code(code: str) -> tuple[bool, str]:
    """Return (is_safe, reason) for the generated code."""
    for token in BANNED_TOKENS:
        if token in code:
            return False, f"Banned token found: '{token}'"
    return True, "ok"


# ── Template selector ─────────────────────────────────────────────────────────
def select_template(query: str) -> str:
    """Pick the closest EDA code template for a query."""
    q_lower = query.lower()
    for pattern, code in EDA_TEMPLATES.items():
        if re.search(pattern, q_lower):
            return textwrap.dedent(code).strip()
    return textwrap.dedent(DEFAULT_CODE).strip()


# ── Safe executor ─────────────────────────────────────────────────────────────
def execute_visualization(
    code: str,
    return_base64: bool = True,
) -> dict:
    """
    Execute visualization code safely in a restricted namespace.

    Returns:
        {
            "success":  bool,
            "image_b64": str | None,   # base64 PNG if return_base64=True
            "image_path": str | None,  # saved .png path if return_base64=False
            "error": str | None,
        }
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    safe, reason = is_safe_code(code)
    if not safe:
        return {"success": False, "image_b64": None, "image_path": None,
                "error": f"Security check failed: {reason}"}

    # Restricted execution namespace — allow safe data-science libs only
    import builtins, importlib
    allowed_modules = {"matplotlib", "matplotlib.pyplot", "seaborn", "numpy",
                       "pandas", "scipy", "math", "random"}

    class RestrictedImport:
        def __init__(self, original_import):
            self._orig = original_import
        def __call__(self, name, *args, **kwargs):
            base = name.split(".")[0]
            if base not in allowed_modules:
                raise ImportError(f"Import of '{name}' is not allowed in visualization sandbox.")
            return self._orig(name, *args, **kwargs)

    safe_builtins = dict(vars(builtins))
    safe_builtins["__import__"] = RestrictedImport(builtins.__import__)
    namespace = {"__builtins__": safe_builtins}

    try:
        exec(code, namespace)   # noqa: S102

        buf = io.BytesIO()
        fig = plt.gcf()
        if not fig.get_axes():
            raise ValueError("No plot was generated.")

        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close("all")
        buf.seek(0)

        if return_base64:
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            return {"success": True, "image_b64": img_b64,
                    "image_path": None, "error": None}
        else:
            path = "/tmp/viz_output.png"
            with open(path, "wb") as f:
                f.write(buf.read())
            return {"success": True, "image_b64": None,
                    "image_path": path, "error": None}

    except Exception:
        plt.close("all")
        return {"success": False, "image_b64": None,
                "image_path": None, "error": traceback.format_exc()}


# ── Main entry: query → (code, result) ───────────────────────────────────────
def handle_eda_query(query: str) -> dict:
    """
    Full pipeline: query → template selection → safe execution → image.

    Args:
        query: User's EDA question (Stage 3).

    Returns:
        {
            "query": str,
            "generated_code": str,
            "success": bool,
            "image_b64": str | None,
            "error": str | None,
        }
    """
    code = select_template(query)
    result = execute_visualization(code)
    return {
        "query":          query,
        "generated_code": code,
        "success":        result["success"],
        "image_b64":      result.get("image_b64"),
        "error":          result.get("error"),
    }


# ── Routing helper (used by Streamlit UI) ────────────────────────────────────
def route_query(query: str, stage_num: int) -> Optional[dict]:
    """
    If stage == 3 (EDA), run visualization module; else return None.
    """
    if stage_num == 3:
        return handle_eda_query(query)
    return None


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Show me the distribution of ages in the dataset",
        "Plot a correlation heatmap for all numeric features",
        "I want to see a boxplot to check for outliers",
        "Show class imbalance in the target variable",
        "Create a scatter plot between Age and Fare",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        res = handle_eda_query(q)
        status = "✓ Image generated" if res["success"] else f"✗ {res['error'][:80]}"
        print(f"Status: {status}")
        print(f"Code snippet: {res['generated_code'][:120]}...")
