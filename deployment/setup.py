"""
PART 11 — DEPLOYMENT-READY STRUCTURE
======================================
Creates the full project scaffold, validates all modules,
and generates a deployment-ready package.

Run:  python deployment/setup.py
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Required folder structure ─────────────────────────────────────────────────
FOLDERS = [
    "data",
    "models",
    "rag",
    "classifier",
    "modules",
    "ui",
    "evaluation",
    "deployment",
    "outputs",
    "notebooks",
]

# ── Required files after full pipeline run ────────────────────────────────────
REQUIRED_GENERATED = [
    "data/dataset.csv",
    "data/train.csv",
    "data/val.csv",
    "data/test.csv",
    "data/stage_labeled_dataset.csv",
    "data/small_sample_dataset.csv",
    "models/tfidf_svm_fallback.pkl",
    "classifier/intent_classifier.pkl",
    "outputs/eval_results.csv",
]

SOURCE_FILES = {
    "data/build_dataset.py":              "Part 1 — Dataset creation + Kaggle scraper",
    "data/create_dataset.py":             "Part 1b — Curated base dataset",
    "data/dataset_extra.py":              "Part 1c — Advanced examples",
    "data/label_stages.py":               "Part 2 — Stage labeling",
    "rag/hybrid_retriever.py":            "Part 3 — BM25 + TF-IDF + FAISS hybrid retrieval",
    "classifier/intent_stage_classifier.py": "Part 4 — Stage + intent classifier",
    "models/stage_classifier.py":         "Part 4b — TF-IDF+SVM stage model",
    "models/finetune_codet5.py":          "Part 5 — CodeT5 LoRA fine-tuning",
    "modules/code_generator.py":          "Part 5b — Context-aware code generation",
    "modules/visualization_sandbox.py":   "Part 6 — Subprocess visualization sandbox",
    "modules/workflow.py":                "Part 7 — Workflow guidance + skip detection",
    "modules/conversation.py":            "Part 8 — Multi-turn conversation memory",
    "ui/app.py":                          "Part 9 — Streamlit 4-panel UI",
    "evaluation/benchmark.py":           "Part 10 — Full evaluation benchmark",
    "deployment/setup.py":               "Part 11 — Deployment setup",
}


def check_structure():
    """Verify all required folders and source files exist."""
    print("=== Checking Project Structure ===\n")
    ok = True
    for folder in FOLDERS:
        p = ROOT / folder
        exists = p.exists()
        print(f"  {'✓' if exists else '✗'} {folder}/")
        if not exists:
            ok = False

    print()
    for path, description in SOURCE_FILES.items():
        p = ROOT / path
        exists = p.exists()
        print(f"  {'✓' if exists else '✗'} {path:<45} {description}")
        if not exists:
            ok = False

    print()
    print("=== Checking Generated Files ===\n")
    missing_generated = []
    for path in REQUIRED_GENERATED:
        p = ROOT / path
        exists = p.exists()
        size   = f"({p.stat().st_size / 1024:.1f} KB)" if exists else ""
        print(f"  {'✓' if exists else '○'} {path} {size}")
        if not exists:
            missing_generated.append(path)

    if missing_generated:
        print(f"\n  ⚠ {len(missing_generated)} generated file(s) missing.")
        print("  Run the pipeline steps first (see README.md or run_pipeline.sh)")
    else:
        print("\n  ✓ All generated files present.")

    return ok, missing_generated


def check_imports():
    """Check all module imports work."""
    print("\n=== Checking Module Imports ===\n")
    checks = [
        ("rank_bm25",             "BM25 retrieval",          False),
        ("sklearn",               "Scikit-learn",            True),
        ("numpy",                 "NumPy",                   True),
        ("pandas",                "Pandas",                  True),
        ("matplotlib",            "Matplotlib",              True),
        ("seaborn",               "Seaborn",                 True),
        ("streamlit",             "Streamlit",               True),
        ("transformers",          "HuggingFace Transformers",False),
        ("torch",                 "PyTorch",                 False),
        ("sentence_transformers", "Sentence-BERT",           False),
        ("faiss",                 "FAISS",                   False),
    ]
    for module, name, required in checks:
        try:
            __import__(module)
            print(f"  ✓ {name:<30} ({module})")
        except ImportError:
            label = "✗ REQUIRED" if required else "○ optional"
            print(f"  {label} {name:<30} ({module})")


def print_run_order():
    """Print the exact commands to run in order."""
    print("\n=== Pipeline Run Order ===\n")
    steps = [
        ("1", "Install dependencies",
         "pip install streamlit pandas scikit-learn matplotlib seaborn numpy rank_bm25 joblib"),
        ("2", "Build dataset (701+ examples + train/val/test splits)",
         "python data/build_dataset.py --no-kaggle"),
        ("2a", "Label pipeline stages",
         "python data/label_stages.py"),
        ("3", "Train stage + intent classifiers",
         "python models/stage_classifier.py\npython classifier/intent_stage_classifier.py"),
        ("4", "Test hybrid retrieval",
         "python rag/hybrid_retriever.py"),
        ("5", "Test code generation",
         "python modules/code_generator.py"),
        ("6", "Test visualization sandbox",
         "python modules/visualization_sandbox.py"),
        ("7", "Run full evaluation benchmark",
         "python evaluation/benchmark.py"),
        ("8", "Launch Streamlit app",
         "streamlit run ui/app.py"),
    ]
    for num, title, cmd in steps:
        print(f"  Step {num}: {title}")
        for line in cmd.split("\n"):
            print(f"    $ {line}")
        print()


def generate_requirements_txt():
    """Write the full requirements.txt."""
    content = """# DS Mentor QA System — Requirements
# Core (required)
streamlit>=1.28.0
pandas>=1.5.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
rank_bm25>=0.2.2
joblib>=1.3.0

# Optional — for FAISS dense retrieval (improves retrieval quality)
# faiss-cpu>=1.7.4
# sentence-transformers>=2.2.0

# Optional — for DistilBERT/CodeT5 fine-tuning (needs GPU)
# transformers>=4.30.0
# torch>=2.0.0
# datasets>=2.12.0
# peft>=0.4.0

# Optional — for Kaggle notebook download
# kaggle>=1.5.16
"""
    req_path = ROOT / "requirements.txt"
    req_path.write_text(content)
    print(f"  ✓ requirements.txt updated")


def generate_run_script():
    """Write run_pipeline.sh for one-command setup."""
    content = """#!/bin/bash
# DS Mentor QA System — Full Pipeline Setup
# Usage: bash deployment/run_pipeline.sh

set -e
echo "🎓 DS Mentor QA System — Pipeline Setup"
echo "========================================"

echo ""
echo "Step 1: Installing Python packages..."
pip install -q streamlit pandas scikit-learn matplotlib seaborn numpy rank_bm25 joblib

echo ""
echo "Step 2: Building dataset..."
python data/build_dataset.py --no-kaggle

echo ""
echo "Step 3: Labeling pipeline stages..."
python data/label_stages.py

echo ""
echo "Step 4: Training classifiers..."
python models/stage_classifier.py
python classifier/intent_stage_classifier.py

echo ""
echo "Step 5: Running evaluation..."
python evaluation/benchmark.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Launch app with:"
echo "  streamlit run ui/app.py"
echo "  Then open: http://localhost:8501"
"""
    script_path = ROOT / "deployment" / "run_pipeline.sh"
    script_path.write_text(content)
    script_path.chmod(0o755)
    print(f"  ✓ deployment/run_pipeline.sh written")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DS MENTOR QA — DEPLOYMENT SETUP CHECK")
    print("="*60)

    # Create missing folders
    for folder in FOLDERS:
        (ROOT / folder).mkdir(exist_ok=True)
        init = ROOT / folder / "__init__.py"
        if not init.exists():
            init.touch()

    ok, missing = check_structure()
    check_imports()
    generate_requirements_txt()
    generate_run_script()
    print_run_order()

    print("="*60)
    if not missing:
        print("✅ All systems ready. Launch with: streamlit run ui/app.py")
    else:
        print(f"⚠  Run the pipeline steps above to generate {len(missing)} missing file(s).")
    print("="*60)
