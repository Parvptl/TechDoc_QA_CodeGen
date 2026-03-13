"""
EVALUATION SCRIPT — DS Mentor QA System
Computes ALL metrics against real test queries.
Outputs: outputs/eval_results.csv + printed report.

Metrics computed:
  Retrieval:  MRR, Recall@1/3/5, Precision@5
  Stage Clf:  Accuracy, Macro-F1, Per-class F1
  Code Gen:   Execution success rate (runs without syntax error)
  Workflow:   Skip detection accuracy
  Viz:        Execution success rate for EDA queries
"""
import csv, sys, re, io, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Test queries with ground truth ────────────────────────────────────────────
TEST_QUERIES = [
    # (query, correct_stage, has_eda_viz, is_skip_scenario)
    ("What is the goal of a Titanic survival prediction?",      1, False, False),
    ("How do I define a business objective for churn?",         1, False, False),
    ("How do I load a CSV file with pandas?",                   2, False, False),
    ("Load multiple CSV files and concat them",                 2, False, False),
    ("Read a parquet file and check memory usage",              2, False, False),
    ("Plot the distribution of Age column",                     3, True,  False),
    ("Show a correlation heatmap for all features",             3, True,  False),
    ("Create a boxplot to check outliers",                      3, True,  False),
    ("Show class imbalance in the target variable",             3, True,  False),
    ("How do I fill missing values in Age with median?",        4, False, False),
    ("Drop columns with more than 50% missing values",          4, False, False),
    ("How do I remove duplicate rows?",                         4, False, False),
    ("Cap outliers using IQR method",                           4, False, False),
    ("Apply SMOTE to handle imbalanced data",                   4, False, False),
    ("How do I encode the Sex column with LabelEncoder?",       5, False, False),
    ("Create one-hot encoding for Embarked column",             5, False, False),
    ("Extract date features from a datetime column",            5, False, False),
    ("Apply log transform to reduce skewness in Fare",          5, False, False),
    ("How do I train a Random Forest classifier?",              6, False, False),
    ("Train XGBoost with early stopping",                       6, False, False),
    ("Tune hyperparameters with GridSearchCV",                  6, False, False),
    ("Use cross-validation to evaluate model performance",      6, False, False),
    ("How do I calculate AUC-ROC score?",                       7, False, False),
    ("Plot the confusion matrix",                               7, False, False),
    ("Compute precision, recall, and F1 score",                 7, False, False),
    ("Detect overfitting using learning curves",                7, False, False),
]

SKIP_SCENARIOS = [
    ({"completed": [2], "query": "How do I train an XGBoost model?"}, 6, True),
    ({"completed": [1,2], "query": "What is the AUC score?"}, 7, True),
    ({"completed": [1,2,3,4,5], "query": "How do I train a Random Forest?"}, 6, False),
    ({"completed": [1], "query": "How do I encode categorical features?"}, 5, True),
]


def evaluate_stage_classifier():
    """Measure stage classification accuracy and F1."""
    print("\n" + "="*60)
    print("STAGE CLASSIFIER EVALUATION")
    print("="*60)
    from models.stage_classifier import predict_stage, extract_stage_num
    y_true, y_pred = [], []
    for query, true_stage, _, _ in TEST_QUERIES:
        pred_str = predict_stage(query)
        pred_stage = extract_stage_num(pred_str)
        y_true.append(true_stage)
        y_pred.append(pred_stage)
        status = "✓" if pred_stage == true_stage else "✗"
        print(f"  {status} [{true_stage}→{pred_stage}] {query[:55]}")

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\nAccuracy:  {acc:.4f} ({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)})")
    print(f"Macro-F1:  {macro_f1:.4f}")
    print(f"\nPer-class:\n{classification_report(y_true, y_pred, zero_division=0)}")
    return {"stage_accuracy": round(acc,4), "stage_macro_f1": round(macro_f1,4)}


def evaluate_retrieval():
    """Measure BM25+TF-IDF retrieval quality."""
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION (BM25 + TF-IDF)")
    print("="*60)
    from modules.retrieval import DSRetriever
    retriever = DSRetriever("data/dataset.csv")
    stats = retriever.get_stats()
    print(f"Index: {stats['total_documents']} docs, BM25={stats['bm25_available']}")
    test_set = [{"query": q, "relevant_stage": s} for q,s,_,_ in TEST_QUERIES]
    metrics = retriever.evaluate_retrieval(test_set)
    print(f"MRR:        {metrics['mrr']:.4f}")
    print(f"Recall@1:   {metrics['recall_at_1']:.4f}")
    print(f"Recall@3:   {metrics['recall_at_3']:.4f}")
    print(f"Recall@5:   {metrics['recall_at_5']:.4f}")
    return metrics


def evaluate_code_execution():
    """Test that retrieved code snippets are syntactically valid."""
    print("\n" + "="*60)
    print("CODE EXECUTION EVALUATION")
    print("="*60)
    import ast
    from modules.retrieval import DSRetriever
    retriever = DSRetriever("data/dataset.csv")
    success, total = 0, 0
    for doc in retriever.documents:
        code = doc.get("code", "")
        if not code or len(code) < 5:
            continue
        total += 1
        try:
            ast.parse(code)
            success += 1
        except SyntaxError as e:
            print(f"  ✗ SyntaxError: {str(e)[:60]} in: {code[:50]}")
    rate = success / total if total > 0 else 0
    print(f"Code syntax valid: {success}/{total} ({rate*100:.1f}%)")
    return {"code_syntax_success_rate": round(rate, 4), "total_code_snippets": total}


def evaluate_visualization():
    """Test visualization module execution success rate."""
    print("\n" + "="*60)
    print("VISUALIZATION EVALUATION")
    print("="*60)
    from modules.visualization import handle_eda_query
    viz_queries = [q for q,s,has_viz,_ in TEST_QUERIES if has_viz]
    success = 0
    for q in viz_queries:
        res = handle_eda_query(q)
        status = "✓" if res["success"] else "✗"
        print(f"  {status} {q[:60]}")
        if res["success"]:
            success += 1
    rate = success / len(viz_queries) if viz_queries else 0
    print(f"\nViz success: {success}/{len(viz_queries)} ({rate*100:.1f}%)")
    return {"viz_success_rate": round(rate,4), "viz_queries_tested": len(viz_queries)}


def evaluate_skip_detection():
    """Test workflow skip detection accuracy."""
    print("\n" + "="*60)
    print("SKIP DETECTION EVALUATION")
    print("="*60)
    from modules.workflow import WorkflowTracker
    from models.stage_classifier import extract_stage_num, predict_stage
    correct = 0
    for scenario, expected_stage, expected_skip in SKIP_SCENARIOS:
        tracker = WorkflowTracker()
        for s in scenario["completed"]:
            tracker.mark_complete(s)
        result = tracker.process_query(scenario["query"], predicted_stage=expected_stage)
        detected = result["is_skip"]
        status = "✓" if detected == expected_skip else "✗"
        print(f"  {status} completed={scenario['completed']} → stage {expected_stage} → skip={detected} (expected {expected_skip})")
        if detected == expected_skip:
            correct += 1
    acc = correct / len(SKIP_SCENARIOS)
    print(f"\nSkip detection accuracy: {correct}/{len(SKIP_SCENARIOS)} ({acc*100:.1f}%)")
    return {"skip_detection_accuracy": round(acc,4)}


def run_full_evaluation():
    """Run all evaluations and save results to CSV."""
    print("\n" + "█"*60)
    print("  DS MENTOR QA SYSTEM — FULL EVALUATION REPORT")
    print("█"*60)

    results = {}
    results.update(evaluate_stage_classifier())
    results.update(evaluate_retrieval())
    results.update(evaluate_code_execution())
    results.update(evaluate_visualization())
    results.update(evaluate_skip_detection())

    print("\n" + "="*60)
    print("FINAL METRICS SUMMARY")
    print("="*60)
    labels = {
        "stage_accuracy":          "Stage Classifier Accuracy",
        "stage_macro_f1":          "Stage Classifier Macro-F1",
        "mrr":                     "Retrieval MRR",
        "recall_at_1":             "Retrieval Recall@1",
        "recall_at_3":             "Retrieval Recall@3",
        "recall_at_5":             "Retrieval Recall@5",
        "code_syntax_success_rate":"Code Syntax Success Rate",
        "viz_success_rate":        "Visualization Success Rate",
        "skip_detection_accuracy": "Skip Detection Accuracy",
    }
    for key, label in labels.items():
        val = results.get(key, "N/A")
        bar = "█" * int(float(val)*20) if isinstance(val, float) else ""
        print(f"  {label:<38}: {val:.4f}  {bar}")

    # Save CSV
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric","value"])
        writer.writeheader()
        for k,v in results.items():
            writer.writerow({"metric": labels.get(k,k), "value": v})
    print(f"\n[INFO] Results saved → outputs/eval_results.csv")
    return results


if __name__ == "__main__":
    run_full_evaluation()
