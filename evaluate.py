"""
EVALUATION SCRIPT — DS Mentor QA System (Final)
Run: python evaluate.py
"""
import csv, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

TEST_QUERIES = [
    ("What is the goal of a Titanic survival prediction?",      1),
    ("How do I define a business objective for churn?",         1),
    ("What is a null model and why is it needed?",              1),
    ("How do I load a CSV file with pandas?",                   2),
    ("Load multiple CSV files and concat them",                 2),
    ("Read a parquet file and check memory usage",              2),
    ("Plot the distribution of Age column",                     3),
    ("Show a correlation heatmap for all features",             3),
    ("Create a boxplot to check for outliers",                  3),
    ("Show class imbalance in the target variable",             3),
    ("How do I fill missing values in Age with median?",        4),
    ("Drop columns with more than 50% missing values",          4),
    ("How do I remove duplicate rows?",                         4),
    ("Cap outliers using IQR method",                           4),
    ("Apply SMOTE to handle imbalanced data",                   4),
    ("How do I encode the Sex column with LabelEncoder?",       5),
    ("Create one-hot encoding for Embarked column",             5),
    ("Extract date features from a datetime column",            5),
    ("Apply log transform to reduce skewness in Fare",          5),
    ("How do I train a Random Forest classifier?",              6),
    ("Train XGBoost with early stopping",                       6),
    ("Tune hyperparameters with GridSearchCV",                  6),
    ("Use cross-validation to evaluate model performance",      6),
    ("How do I calculate AUC-ROC score?",                       7),
    ("Plot the confusion matrix",                               7),
    ("Detect overfitting using learning curves",                7),
]
EDA_QUERIES = [q for q,s in TEST_QUERIES if s==3]
SKIP_SCENARIOS = [
    ({"completed":[2],"query":"How do I train an XGBoost model?"},6,True),
    ({"completed":[1,2],"query":"What is the AUC score?"},7,True),
    ({"completed":[1,2,3,4,5],"query":"How do I train a Random Forest?"},6,False),
    ({"completed":[1],"query":"How do I encode categorical features?"},5,True),
]
CONV_SCENARIO=[
    ("How do I load 'titanic.csv' with pandas?",False),
    ("Now show the distribution of 'Age'",True),
    ("How do I fill missing values in it?",True),
    ("And do the same for 'Fare'",True),
    ("How do I evaluate the model?",True),
]
CODE_GEN_QUERIES=[
    ("Fill missing values in 'Age' using median",4),
    ("Train a Random Forest with 100 estimators",6),
    ("Compute AUC and plot confusion matrix",7),
    ("Load 'data/train.csv' with pandas",2),
    ("Plot distribution of 'Fare' column",3),
]

def evaluate_stage_classifier():
    print("\n"+"="*60+"\n1. STAGE CLASSIFIER\n"+"="*60)
    from models.stage_classifier import predict_stage,extract_stage_num
    y_true,y_pred=[],[]
    for q,s in TEST_QUERIES:
        p=extract_stage_num(predict_stage(q)); y_true.append(s); y_pred.append(p)
        print(f"  {'v' if p==s else 'x'} [{s}->{p}] {q[:55]}")
    from sklearn.metrics import accuracy_score,f1_score,classification_report
    acc=accuracy_score(y_true,y_pred); f1=f1_score(y_true,y_pred,average="macro",zero_division=0)
    print(f"\n  Accuracy:{acc:.4f} Macro-F1:{f1:.4f}")
    print(classification_report(y_true,y_pred,zero_division=0))
    return {"stage_accuracy":round(acc,4),"stage_macro_f1":round(f1,4)}

def evaluate_retrieval():
    print("\n"+"="*60+"\n2. RETRIEVAL\n"+"="*60)
    from modules.retrieval import DSRetriever
    r=DSRetriever("data/dataset.csv"); s=r.get_stats()
    print(f"  Docs:{s['total_documents']} BM25:{s['bm25_available']} Vocab:{s['tfidf_vocab_size']:,}")
    t0=time.time(); m=r.evaluate_retrieval([{"query":q,"relevant_stage":s} for q,s in TEST_QUERIES])
    lat=(time.time()-t0)/len(TEST_QUERIES)*1000
    print(f"  MRR={m['mrr']:.4f} R@1={m['recall_at_1']:.4f} R@3={m['recall_at_3']:.4f} R@5={m['recall_at_5']:.4f}")
    print(f"  Latency:{lat:.1f}ms/query")
    return {**m,"retrieval_latency_ms":round(lat,2)}

def evaluate_code_generation():
    print("\n"+"="*60+"\n3. CODE GENERATION\n"+"="*60)
    from modules.code_generator import generate_code,validate_code_syntax
    from modules.retrieval import DSRetriever
    r=DSRetriever("data/dataset.csv")
    syn_ok=sum(1 for d in r.documents if validate_code_syntax(d.get("code",""))["valid"])
    total=len(r.documents)
    print(f"  Dataset syntax valid:{syn_ok}/{total} ({syn_ok/total*100:.1f}%)")
    correct=0
    for q,stage in CODE_GEN_QUERIES:
        res=generate_code(q,stage); code=res["code"]; ctx=res["context"]
        valid=validate_code_syntax(code)["valid"]
        col_ok=not ctx["columns"] or any(c in code for c in ctx["columns"])
        ok=valid and col_ok; correct+=1 if ok else 0
        print(f"  {'v' if ok else 'x'} [{res['method'][:3]}] {q[:55]}")
    rate=correct/len(CODE_GEN_QUERIES)
    print(f"  Context accuracy:{correct}/{len(CODE_GEN_QUERIES)} ({rate*100:.1f}%)")
    return {"code_syntax_success_rate":round(syn_ok/total,4),"code_context_accuracy":round(rate,4),"total_snippets":total}

def evaluate_visualization():
    print("\n"+"="*60+"\n4. VISUALIZATION\n"+"="*60)
    from modules.visualization import handle_eda_query
    ok,lats=0,[]
    for q in EDA_QUERIES:
        t0=time.time(); res=handle_eda_query(q); lats.append(time.time()-t0)
        ok+=1 if res["success"] else 0
        print(f"  {'v' if res['success'] else 'x'} ({lats[-1]:.2f}s) {q[:60]}")
    rate=ok/len(EDA_QUERIES)
    print(f"  Success:{ok}/{len(EDA_QUERIES)} Avg:{np.mean(lats):.2f}s")
    return {"viz_success_rate":round(rate,4),"viz_avg_latency_s":round(float(np.mean(lats)),2)}

def evaluate_skip_detection():
    print("\n"+"="*60+"\n5. SKIP DETECTION\n"+"="*60)
    from modules.workflow import WorkflowTracker
    correct=0
    for sc,exp_s,exp_sk in SKIP_SCENARIOS:
        t=WorkflowTracker()
        for s in sc["completed"]: t.mark_complete(s)
        res=t.process_query(sc["query"],predicted_stage=exp_s)
        ok=res["is_skip"]==exp_sk; correct+=1 if ok else 0
        print(f"  {'v' if ok else 'x'} done={sc['completed']} S{exp_s} skip={res['is_skip']} exp={exp_sk}")
    acc=correct/len(SKIP_SCENARIOS)
    print(f"  Accuracy:{correct}/{len(SKIP_SCENARIOS)} ({acc*100:.1f}%)")
    return {"skip_detection_accuracy":round(acc,4)}

def evaluate_conversation():
    print("\n"+"="*60+"\n6. MULTI-TURN CONVERSATION\n"+"="*60)
    from modules.conversation import ConversationManager
    mgr=ConversationManager(); fu_ok=res_ok=total=0
    for q,exp_fu in CONV_SCENARIO:
        r=mgr.process_turn(q)
        ok_fu=r["is_followup"]==exp_fu
        ok_res=(r["enriched_query"]!=q)==(exp_fu and bool(r["references"]))
        fu_ok+=1 if ok_fu else 0; res_ok+=1 if ok_res else 0; total+=1
        resolved=f" -> {r['enriched_query'][:50]}" if r["enriched_query"]!=q else ""
        print(f"  {'v' if ok_fu else 'x'} [fu={r['is_followup']}] {q[:45]}{resolved}")
        mgr.record_turn(q,1,"Test","Ans","# code")
    fu_acc=fu_ok/total; res_acc=res_ok/total
    print(f"  Follow-up:{fu_ok}/{total} ({fu_acc*100:.1f}%) Resolution:{res_ok}/{total} ({res_acc*100:.1f}%)")
    return {"conv_followup_accuracy":round(fu_acc,4),"conv_resolution_accuracy":round(res_acc,4)}

def run_full_evaluation():
    print("\n"+"#"*60+"\n  DS MENTOR QA — COMPLETE EVALUATION\n"+"#"*60)
    results={}
    results.update(evaluate_stage_classifier())
    results.update(evaluate_retrieval())
    results.update(evaluate_code_generation())
    results.update(evaluate_visualization())
    results.update(evaluate_skip_detection())
    results.update(evaluate_conversation())
    LABELS={
        "stage_accuracy":"Stage Classifier Accuracy",
        "stage_macro_f1":"Stage Classifier Macro-F1",
        "mrr":"Retrieval MRR",
        "recall_at_1":"Retrieval Recall@1",
        "recall_at_3":"Retrieval Recall@3",
        "recall_at_5":"Retrieval Recall@5",
        "retrieval_latency_ms":"Retrieval Latency (ms)",
        "code_syntax_success_rate":"Code Syntax Success Rate",
        "code_context_accuracy":"Code Context Accuracy",
        "viz_success_rate":"Visualization Success Rate",
        "viz_avg_latency_s":"Visualization Latency (s)",
        "skip_detection_accuracy":"Skip Detection Accuracy",
        "conv_followup_accuracy":"Conv Follow-up Detection",
        "conv_resolution_accuracy":"Conv Reference Resolution",
    }
    print("\n"+"="*60+"\nFINAL METRICS SUMMARY\n"+"="*60)
    for k,label in LABELS.items():
        v=results.get(k,"N/A")
        bar="█"*int(v*20) if isinstance(v,float) and v<=1.0 else ""
        fmt=f"{v:.4f}" if isinstance(v,float) and v<=1.0 else str(v)
        print(f"  {label:<44}: {fmt}  {bar}")
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/eval_results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["metric","value"]); w.writeheader()
        [w.writerow({"metric":LABELS.get(k,k),"value":v}) for k,v in results.items()]
    md=["# DS Mentor QA — Evaluation Report\n\n","| Metric | Value |\n|--------|-------|\n"]
    for k,label in LABELS.items():
        v=results.get(k,"N/A"); md.append(f"| {label} | {f'{v:.4f}' if isinstance(v,float) else v} |\n")
    with open("outputs/eval_report.md","w") as f: f.writelines(md)
    print("\n[INFO] Saved: outputs/eval_results.csv + outputs/eval_report.md")
    return results

if __name__=="__main__":
    run_full_evaluation()


def evaluate_code_generator():
    """Test context-aware code generation for all 7 stages."""
    print("\n" + "="*60)
    print("CODE GENERATOR EVALUATION")
    print("="*60)
    from modules.code_generator import generate_code, validate_code_syntax
    test_cases = [
        ("Predict 'Survived' with AUC metric",                1),
        ("Load 'train.csv' with pandas",                      2),
        ("Show distribution of 'Age' column",                 3),
        ("Fill missing values in 'Fare' using median",        4),
        ("Encode 'Sex' column with LabelEncoder",             5),
        ("Train Random Forest with 200 estimators",           6),
        ("Compute AUC and plot confusion matrix",             7),
        ("Train XGBoost with early stopping",                 6),
        ("Handle missing values using KNN imputation",        4),
        ("Create one-hot encoding for 'Embarked'",            5),
    ]
    results = []
    for query, stage in test_cases:
        r = generate_code(query, stage)
        v = validate_code_syntax(r["code"])
        ctx_used = any(v for v in r.get("context",{}).values() if v)
        status = "✓" if v["valid"] else "✗"
        method_mark = "📌" if ctx_used else "📄"
        print(f"  {status} {method_mark} [S{stage}] {query[:50]:<50} | {r['method']}")
        results.append({"valid": v["valid"], "ctx_used": ctx_used, "method": r["method"]})
    valid_rate = sum(1 for r in results if r["valid"]) / len(results)
    ctx_rate   = sum(1 for r in results if r["ctx_used"]) / len(results)
    print(f"\nSyntax valid: {sum(1 for r in results if r['valid'])}/{len(results)} ({valid_rate*100:.0f}%)")
    print(f"Context used: {sum(1 for r in results if r['ctx_used'])}/{len(results)} ({ctx_rate*100:.0f}%)")
    return {"codegen_syntax_rate": round(valid_rate,4), "codegen_context_rate": round(ctx_rate,4)}


def evaluate_conversation():
    """Test multi-turn conversation and pronoun resolution."""
    print("\n" + "="*60)
    print("MULTI-TURN CONVERSATION EVALUATION")
    print("="*60)
    from modules.conversation import ConversationManager

    test_scenarios = [
        {
            "setup": [
                ("Load 'titanic.csv' with pandas", 2, "Data Loading", "use pd.read_csv", "df=pd.read_csv('titanic.csv')"),
                ("Show distribution of 'Age'", 3, "EDA", "use histplot", "sns.histplot(df['Age'])"),
            ],
            "followup": "How do I fill missing values in it?",
            "expect_followup": True,
            "expect_in_resolved": "Age",
        },
        {
            "setup": [
                ("Train a Random Forest model", 6, "Modeling", "use RandomForestClassifier", "model=RFC().fit(X,y)"),
            ],
            "followup": "How do I evaluate the model?",
            "expect_followup": True,
            "expect_in_resolved": "Random Forest",
        },
        {
            "setup": [],
            "followup": "What is the goal of a Titanic survival task?",
            "expect_followup": False,
            "expect_in_resolved": None,
        },
        {
            "setup": [
                ("Load 'data.csv'", 2, "Data Loading", "use pd.read_csv", "df=pd.read_csv('data.csv')"),
            ],
            "followup": "And what about the test data?",
            "expect_followup": True,
            "expect_in_resolved": None,
        },
    ]

    correct = 0
    for i, scenario in enumerate(test_scenarios):
        mgr = ConversationManager()
        for q, s, sn, ans, code in scenario["setup"]:
            mgr.record_turn(q, s, sn, ans, code)
        result = mgr.process_turn(scenario["followup"])
        followup_ok = result["is_followup"] == scenario["expect_followup"]
        resolve_ok  = True
        if scenario["expect_in_resolved"]:
            resolve_ok = scenario["expect_in_resolved"] in result["enriched_query"]
        ok = followup_ok and resolve_ok
        if ok: correct += 1
        status = "✓" if ok else "✗"
        print(f"  {status} Scenario {i+1}: {scenario['followup'][:55]}")
        if not followup_ok:
            print(f"       Expected is_followup={scenario['expect_followup']}, got {result['is_followup']}")
        if not resolve_ok:
            print(f"       Expected '{scenario['expect_in_resolved']}' in enriched query: {result['enriched_query'][:80]}")

    acc = correct / len(test_scenarios)
    print(f"\nConversation accuracy: {correct}/{len(test_scenarios)} ({acc*100:.0f}%)")
    return {"conversation_accuracy": round(acc, 4)}


def run_full_evaluation_v2():
    """Run complete evaluation including new modules."""
    print("\n" + "█"*60)
    print("  DS MENTOR QA SYSTEM — COMPLETE EVALUATION REPORT v2")
    print("█"*60)

    results = {}
    results.update(evaluate_stage_classifier())
    results.update(evaluate_retrieval())
    results.update(evaluate_code_generation())
    results.update(evaluate_visualization())
    results.update(evaluate_skip_detection())
    results.update(evaluate_code_generator())
    results.update(evaluate_conversation())

    print("\n" + "="*60)
    print("FINAL METRICS SUMMARY")
    print("="*60)
    labels = {
        "stage_accuracy":           "Stage Classifier Accuracy",
        "stage_macro_f1":           "Stage Classifier Macro-F1",
        "mrr":                      "Retrieval MRR",
        "recall_at_1":              "Retrieval Recall@1",
        "recall_at_3":              "Retrieval Recall@3",
        "recall_at_5":              "Retrieval Recall@5",
        "code_syntax_success_rate": "Dataset Code Syntax Rate",
        "viz_success_rate":         "Visualization Success Rate",
        "skip_detection_accuracy":  "Skip Detection Accuracy",
        "codegen_syntax_rate":      "Code Generator Syntax Rate",
        "codegen_context_rate":     "Code Generator Context Rate",
        "conversation_accuracy":    "Multi-turn Conversation Accuracy",
    }
    for key, label in labels.items():
        val = results.get(key, "N/A")
        if isinstance(val, float):
            bar = "█" * int(val * 20)
            print(f"  {label:<42}: {val:.4f}  {bar}")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric","value"])
        writer.writeheader()
        for k,v in results.items():
            writer.writerow({"metric": labels.get(k,k), "value": v})
    print(f"\n[INFO] Results saved → outputs/eval_results.csv")
    return results


if __name__ == "__main__":
    run_full_evaluation_v2()
