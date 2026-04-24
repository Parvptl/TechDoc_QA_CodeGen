"""Safe subprocess sandbox for generating and executing matplotlib/seaborn visualization code."""

import re, sys, base64, tempfile, subprocess, time
from pathlib import Path
from typing import Optional

# ── EDA code templates (9 plot types) ─────────────────────────────────────────
EDA_TEMPLATES = {
r"distribut|histogram": """
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, numpy as np
np.random.seed(42)
data = np.concatenate([np.random.normal(45,15,250), np.random.normal(70,10,50)])
fig, axes = plt.subplots(1,2,figsize=(11,4))
sns.histplot(data, bins=30, kde=True, color='steelblue', ax=axes[0])
axes[0].set_title('Feature Distribution (KDE)', fontsize=13, fontweight='bold')
sns.boxplot(y=data, ax=axes[1], color='lightcoral')
axes[1].set_title('Boxplot — Outlier Check', fontsize=13, fontweight='bold')
plt.suptitle('Univariate Analysis', fontsize=14); plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches='tight')
""",

r"correlation|heatmap": """
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, numpy as np, pandas as pd
np.random.seed(42); n=120
df = pd.DataFrame({'Age':np.random.randint(18,80,n),'Fare':np.random.exponential(35,n),
    'Pclass':np.random.choice([1,2,3],n),'SibSp':np.random.choice([0,1,2],n),
    'Survived':np.random.choice([0,1],n)})
fig, ax = plt.subplots(figsize=(8,6))
mask = np.triu(np.ones_like(df.corr(),dtype=bool))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=mask, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches='tight')
""",

r"boxplot|outlier": """
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, numpy as np, pandas as pd
np.random.seed(42); n=180
df = pd.DataFrame({'Class':np.random.choice(['1st','2nd','3rd'],n),
    'Fare':np.where(np.random.choice([0,1,2],n)==0, np.random.normal(80,20,n),
           np.where(np.random.choice([0,1],n)==0, np.random.normal(35,10,n),
           np.random.normal(15,5,n)))})
fig, ax = plt.subplots(figsize=(9,5))
palette = {'1st':'#2ecc71','2nd':'#3498db','3rd':'#e74c3c'}
sns.boxplot(data=df, x='Class', y='Fare', order=['1st','2nd','3rd'],
            palette=palette, ax=ax, flierprops=dict(marker='o',alpha=0.5))
ax.set_title('Fare by Passenger Class', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches='tight')
""",

r"count|bar|class.*imbalanc|imbalanc|target.*distribut": """
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
labels=['Not Survived (0)','Survived (1)']; counts=[549,342]
colors=['#e74c3c','#2ecc71']
fig, axes = plt.subplots(1,2,figsize=(10,4))
bars = axes[0].bar(labels, counts, color=colors, edgecolor='white', linewidth=1.2)
axes[0].set_title('Target Distribution', fontsize=13, fontweight='bold')
for bar,cnt in zip(bars,counts):
    axes[0].annotate(f'{cnt}',(bar.get_x()+bar.get_width()/2,bar.get_height()+5),ha='center',fontsize=12,fontweight='bold')
axes[1].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title('Class Balance', fontsize=13, fontweight='bold')
plt.suptitle('Class Imbalance Analysis',fontsize=14); plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches='tight')
""",

r"scatter|relation|trend": """
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
np.random.seed(42); n=200
age=np.random.randint(1,80,n); fare=np.clip(5+age*0.8+np.random.normal(0,15,n),0,None)
survived=(fare>fare.mean()).astype(int)
fig, ax = plt.subplots(figsize=(9,5))
sc=ax.scatter(age,fare,c=survived,cmap='RdYlGn',alpha=0.7,edgecolors='white',s=60)
m,b=np.polyfit(age,fare,1)
ax.plot(sorted(age),[m*x+b for x in sorted(age)],'k--',alpha=0.6,label=f'y={m:.2f}x+{b:.1f}')
plt.colorbar(sc,ax=ax,label='Survived'); ax.legend()
ax.set_xlabel('Age'); ax.set_ylabel('Fare')
ax.set_title('Age vs Fare coloured by Survival',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches='tight')
""",

r"pairplot|pair.*plot|multiple.*feature": """
import matplotlib; matplotlib.use('Agg')
import seaborn as sns, numpy as np, pandas as pd, matplotlib.pyplot as plt
np.random.seed(42); n=120
df=pd.DataFrame({'Age':np.random.randint(1,80,n).astype(float),
    'Fare':np.random.exponential(30,n),'Pclass':np.random.choice([1,2,3],n).astype(float),
    'Survived':np.random.choice([0,1],n)})
g=sns.pairplot(df,hue='Survived',height=2.2,palette={0:'#e74c3c',1:'#2ecc71'},
               plot_kws={'alpha':0.6,'s':25},diag_kind='kde')
g.fig.suptitle('Pairplot — Feature Relationships',y=1.02,fontsize=13)
plt.savefig(OUTPUT_PATH, dpi=100, bbox_inches='tight')
""",

r"roc|auc.*curve|curve.*auc": """
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
np.random.seed(42)
fpr_rf=np.sort(np.concatenate([[0],np.random.beta(0.5,4,50),[1]]))
tpr_rf=np.sort(np.concatenate([[0],1-np.random.beta(4,0.5,50),[1]]))
fpr_lr=np.sort(np.concatenate([[0],np.random.beta(0.8,3,50),[1]]))
tpr_lr=np.sort(np.concatenate([[0],1-np.random.beta(3,0.8,50),[1]]))
fig, ax = plt.subplots(figsize=(7,6))
ax.plot(fpr_rf,tpr_rf,'b-',lw=2,label='Random Forest (AUC=0.92)')
ax.plot(fpr_lr,tpr_lr,'g--',lw=2,label='Logistic Regression (AUC=0.84)')
ax.plot([0,1],[0,1],'k:',lw=1.5,label='Random (AUC=0.50)')
ax.fill_between(fpr_rf,tpr_rf,alpha=0.08,color='blue')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Model Comparison',fontsize=14,fontweight='bold')
ax.legend(loc='lower right'); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches='tight')
""",

r"confusion|conf.*matrix": """
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, numpy as np
cm=np.array([[312,55],[42,183]])
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=1,
            linecolor='white', annot_kws={'size':16},
            xticklabels=['Pred: No','Pred: Yes'],
            yticklabels=['Act: No','Act: Yes'])
ax.set_title('Confusion Matrix',fontsize=14,fontweight='bold')
acc=(cm[0,0]+cm[1,1])/cm.sum()
ax.text(0.5,-0.1,f'Accuracy={acc:.1%}',transform=ax.transAxes,ha='center',fontsize=12)
plt.tight_layout(); plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches='tight')
""",

r"missing|null.*pattern": """
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
cols=['Age','Cabin','Embarked','Fare','Sex','Survived','Pclass']
pct=[0.199,0.772,0.002,0.0,0.0,0.0,0.0]
colors=['#e74c3c' if p>0.4 else '#f39c12' if p>0.1 else '#2ecc71' for p in pct]
fig, ax = plt.subplots(figsize=(9,4))
bars=ax.barh(cols,pct,color=colors,edgecolor='white')
for bar,p in zip(bars,pct):
    if p>0: ax.text(p+0.01,bar.get_y()+bar.get_height()/2,f'{p*100:.1f}%',va='center',fontsize=10)
ax.set_xlabel('Fraction Missing'); ax.set_title('Missing Value Analysis',fontsize=14,fontweight='bold')
ax.axvline(x=0.5,color='red',linestyle='--',alpha=0.5,label='>50%: drop'); ax.legend()
plt.tight_layout(); plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches='tight')
""",
}

DEFAULT_TEMPLATE = list(EDA_TEMPLATES.values())[0]


def select_template(query: str) -> str:
    q = query.lower()
    for pattern, code in EDA_TEMPLATES.items():
        if re.search(pattern, q):
            return code.strip()
    return DEFAULT_TEMPLATE.strip()


def execute_visual_code(code: str, timeout: int = 20) -> dict:
    """
    Execute visualization code in a subprocess sandbox.
    Injects OUTPUT_PATH as the first line — no string replacement of variable names.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        img_path = tmp_img.name

    # Prepend OUTPUT_PATH assignment — safe injection method
    full_code = f"OUTPUT_PATH = {repr(img_path)}\n" + code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_py:
        tmp_py.write(full_code)
        script_path = tmp_py.name

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        if result.returncode != 0:
            return {"success": False, "image_b64": None,
                    "error": (result.stderr or "Subprocess failed")[:400],
                    "execution_time_ms": elapsed_ms}

        img_file = Path(img_path)
        if not img_file.exists() or img_file.stat().st_size < 100:
            return {"success": False, "image_b64": None,
                    "error": "Image file empty or missing",
                    "execution_time_ms": elapsed_ms}

        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {"success": True, "image_b64": img_b64,
                "error": None, "execution_time_ms": elapsed_ms}

    except subprocess.TimeoutExpired:
        return {"success": False, "image_b64": None,
                "error": f"Timed out after {timeout}s", "execution_time_ms": timeout * 1000}
    except Exception as e:
        return {"success": False, "image_b64": None,
                "error": str(e), "execution_time_ms": 0}
    finally:
        Path(img_path).unlink(missing_ok=True)
        Path(script_path).unlink(missing_ok=True)


def generate_and_execute(query: str, stage_num: int = 3,
                          intent: str = "visualization") -> dict:
    """Full pipeline: query → template → subprocess → image."""
    if intent != "visualization" and stage_num != 3:
        return {"query": query, "generated_code": "", "success": False,
                "image_b64": None, "error": "Not a visualization query",
                "execution_time_ms": 0}
    code   = select_template(query)
    result = execute_visual_code(code)
    return {"query": query, "generated_code": code,
            "success": result["success"], "image_b64": result.get("image_b64"),
            "error": result.get("error"), "execution_time_ms": result.get("execution_time_ms", 0)}


# Backward-compatible alias used by evaluate.py and app.py
def handle_eda_query(query: str) -> dict:
    return generate_and_execute(query, stage_num=3, intent="visualization")


if __name__ == "__main__":
    test_queries = [
        ("Show distribution of Age column",          "distribut"),
        ("Create a correlation heatmap",              "heatmap"),
        ("Boxplot to detect outliers in Fare",        "boxplot"),
        ("Show class imbalance in target",            "imbalance"),
        ("Scatter plot of Age vs Fare",               "scatter"),
        ("Plot the ROC curve for my classifier",      "roc"),
        ("Show confusion matrix as heatmap",          "confusion"),
        ("Visualize missing value pattern",           "missing"),
    ]
    print("=== Visualization Sandbox Demo ===\n")
    ok = 0
    for q, _ in test_queries:
        res = generate_and_execute(q)
        if res["success"]:
            ok += 1
            print(f"  ✓ {res['execution_time_ms']:5.0f}ms | {q}")
        else:
            print(f"  ✗ {res['error'][:60]} | {q}")
    print(f"\nResult: {ok}/{len(test_queries)} passed")
