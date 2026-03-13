"""
PART 6 — STREAMLIT APP (FULLY UPGRADED)
RAG-powered: retrieves real answers from curated dataset.
Four-panel UI + session tracking + skip detection + live evaluation.
"""
import sys, os, base64, textwrap
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="DS Mentor QA", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

# ── Cached resource loading ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval index...")
def load_retriever():
    from modules.retrieval import DSRetriever
    return DSRetriever(str(ROOT / "data/dataset.csv"))

@st.cache_resource(show_spinner="Loading stage classifier...")
def load_classifier():
    from models.stage_classifier import predict_stage_with_confidence, extract_stage_num
    return predict_stage_with_confidence, extract_stage_num

# ── Stage metadata ────────────────────────────────────────────────────────
STAGE_NAMES = {
    1:"Problem Understanding", 2:"Data Loading",
    3:"Exploratory Data Analysis", 4:"Preprocessing",
    5:"Feature Engineering", 6:"Modeling", 7:"Evaluation",
}
STAGE_COLORS = {
    1:"#6c5ce7", 2:"#0984e3", 3:"#00b894",
    4:"#fdcb6e", 5:"#e17055", 6:"#d63031", 7:"#2d3436",
}
STAGE_ICONS = {1:"🎯",2:"📂",3:"📊",4:"🧹",5:"⚙️",6:"🤖",7:"📈"}

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stage-badge{display:inline-block;padding:4px 14px;border-radius:20px;color:white;
             font-weight:700;font-size:.85em;margin-bottom:8px}
.panel-title{font-size:.85em;font-weight:800;text-transform:uppercase;
             letter-spacing:.1em;color:#636e72;margin-bottom:6px}
.warning-box{background:#fff3cd;border-left:5px solid #ffc107;padding:12px 16px;
             border-radius:4px;margin:8px 0}
.success-box{background:#d4edda;border-left:5px solid #28a745;padding:10px 16px;
             border-radius:4px;margin:8px 0}
.retrieval-card{background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;
                padding:10px 14px;margin:4px 0;font-size:.9em}
.metric-box{background:#f0f4ff;border:1px solid #c0cff0;border-radius:6px;
            padding:8px 12px;text-align:center}
</style>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 DS Mentor")
    st.markdown("**Data Science Pipeline QA System**")
    st.caption("Powered by BM25 + TF-IDF Hybrid Retrieval")
    st.divider()

    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.clear(); st.rerun()

    st.markdown("### ⚡ Quick Stage Jump")
    for s, name in STAGE_NAMES.items():
        if st.button(f"{STAGE_ICONS[s]} {s}. {name}", key=f"jump_{s}", use_container_width=True):
            st.session_state["forced_stage"] = s
            st.session_state["preset_query"] = f"Tell me about {name.lower()}"
            st.rerun()

    st.divider()

    # Live session metrics
    if "history" in st.session_state and st.session_state["history"]:
        hist = st.session_state["history"]
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        col1.metric("Queries", len(hist))
        stages_visited = len(set(h["stage"] for h in hist))
        col2.metric("Stages", f"{stages_visited}/7")
        warnings = sum(1 for h in hist if h.get("has_warning"))
        if warnings:
            st.warning(f"⚠️ {warnings} skip warning(s) issued")

# ── Session state init ─────────────────────────────────────────────────────
if "workflow" not in st.session_state:
    from modules.workflow import WorkflowTracker
    st.session_state["workflow"] = WorkflowTracker()
if "history" not in st.session_state:
    st.session_state["history"] = []

workflow = st.session_state["workflow"]

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🎓 Data Science Mentor QA System")
st.caption("Ask any data science question — get grounded explanations, real code, and workflow guidance.")

# ── System status banner ───────────────────────────────────────────────────
with st.expander("🔧 System Status", expanded=False):
    try:
        retriever = load_retriever()
        stats = retriever.get_stats()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📚 Dataset Size", stats["total_documents"])
        c2.metric("🔍 BM25", "✅ Active" if stats["bm25_available"] else "⚠️ Fallback")
        c3.metric("🧠 TF-IDF Vocab", f"{stats['tfidf_vocab_size']:,}")
        c4.metric("🎯 Retrieval Mode", "Hybrid")
        st.success("✅ Retrieval index loaded and ready.")
    except Exception as e:
        st.error(f"❌ Retrieval index error: {e}")
        st.info("Run `python data/create_dataset.py` and `python data/label_stages.py` first.")

st.divider()

# ── Query input ────────────────────────────────────────────────────────────
col_q, col_btn = st.columns([5,1])
with col_q:
    preset = st.session_state.pop("preset_query", "")
    query = st.text_input("Ask a data science question:",
                          value=preset,
                          placeholder="e.g. How do I handle missing values in Age column?",
                          label_visibility="collapsed")
with col_btn:
    submit = st.button("Ask 🔍", use_container_width=True, type="primary")

# ── Example queries ────────────────────────────────────────────────────────
with st.expander("💡 Example queries — click to try"):
    examples = [
        "What is the goal of a Titanic survival prediction task?",
        "How do I load a CSV file with pandas?",
        "Plot a correlation heatmap for all numeric features",
        "How do I handle missing values in the Age column?",
        "How do I encode the Sex column with LabelEncoder?",
        "Train a Random Forest classifier with cross-validation",
        "How do I plot the ROC curve and compute AUC?",
        "Show class imbalance in the target variable",
        "How do I tune hyperparameters with GridSearchCV?",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i%3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["preset_query"] = ex
            st.rerun()

st.divider()

# ── Main response ──────────────────────────────────────────────────────────
if submit and query.strip():
    retriever = load_retriever()
    predict_fn, extract_num = load_classifier()

    # Stage classification
    forced = st.session_state.pop("forced_stage", None)
    if forced:
        stage_num = forced
        stage_label = f"Stage {forced} — {STAGE_NAMES[forced]}"
        confidence = 1.0
    else:
        stage_label, confidence = predict_fn(query)
        stage_num = extract_num(stage_label)

    stage_name = STAGE_NAMES.get(stage_num, "Unknown")
    color = STAGE_COLORS.get(stage_num, "#999")

    # Workflow tracking
    wf = workflow.process_query(query, predicted_stage=stage_num)

    # RAG retrieval
    results = retriever.retrieve(query, top_k=5, predicted_stage=stage_num)
    top_result = results[0] if results else None

    # Log to history
    st.session_state["history"].append({
        "query": query, "stage": stage_num, "stage_name": stage_name,
        "has_warning": bool(wf["warning"]), "confidence": confidence,
    })

    # ── Stage badge + confidence ───────────────────────────────────────
    badge_col, conf_col = st.columns([3,1])
    with badge_col:
        st.markdown(
            f'<span class="stage-badge" style="background:{color}">'
            f'{STAGE_ICONS.get(stage_num,"")} Stage {stage_num} — {stage_name}'
            f'</span>', unsafe_allow_html=True)
    with conf_col:
        st.metric("Classifier Confidence", f"{confidence:.1%}")

    # ── Warning or success ────────────────────────────────────────────
    if wf["warning"]:
        st.markdown(f'<div class="warning-box">{wf["warning"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ Good workflow progression!</div>',
                    unsafe_allow_html=True)

    # ── Four panels ───────────────────────────────────────────────────
    top_left, top_right = st.columns(2)
    bot_left, bot_right = st.columns(2)

    # Panel 1: Explanation (from RAG)
    with top_left:
        st.markdown('<p class="panel-title">💬 Explanation (RAG Retrieved)</p>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            if top_result:
                st.markdown(f"**{top_result['explanation']}**")
                st.caption(f"Source: {top_result.get('source','curated')} · "
                           f"Difficulty: {top_result.get('difficulty','intermediate')} · "
                           f"Retrieval score: {top_result['retrieval_score']:.5f}")
            else:
                st.info("No matching document found. Try rephrasing your question.")

            # Show alternate results
            if len(results) > 1:
                with st.expander(f"📑 See {len(results)-1} more retrieved results"):
                    for r in results[1:]:
                        stage_match = "✓" if int(r["pipeline_stage"]) == stage_num else "○"
                        st.markdown(
                            f'<div class="retrieval-card">'
                            f'<b>{stage_match} [Stage {r["pipeline_stage"]}]</b> '
                            f'{r["explanation"][:100]}... '
                            f'<small>score={r["retrieval_score"]:.5f}</small></div>',
                            unsafe_allow_html=True)

    # Panel 2: Generated Code (from RAG)
    with top_right:
        st.markdown('<p class="panel-title">💻 Retrieved Code</p>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            if top_result:
                st.code(top_result["code"], language="python")
            else:
                st.info("No code found for this query.")

    # Panel 3: Visualization
    with bot_left:
        st.markdown('<p class="panel-title">📊 Visualization</p>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            if stage_num == 3:
                with st.spinner("Generating plot..."):
                    from modules.visualization import handle_eda_query
                    viz = handle_eda_query(query)
                if viz["success"]:
                    st.image(base64.b64decode(viz["image_b64"]), use_container_width=True)
                    with st.expander("🔍 View visualization code"):
                        st.code(viz["generated_code"], language="python")
                else:
                    st.warning(f"Viz error: {str(viz.get('error',''))[:200]}")
            else:
                st.info(
                    f"📊 Visualizations generate for **Stage 3 (EDA)** queries.\n\n"
                    f"This query → Stage {stage_num}: {stage_name}\n\n"
                    f"Try: *'Plot a correlation heatmap for all features'*"
                )

    # Panel 4: Workflow Checklist
    with bot_right:
        st.markdown('<p class="panel-title">📋 Workflow Checklist</p>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            for s, info in wf["checklist"].items():
                icon = "✅" if info["completed"] else "⬜"
                imp  = "🔴" if info["importance"] == "critical" else "🟡"
                curr = " ← **current**" if s == stage_num else ""
                st.markdown(f"{icon} {imp} Stage {s}: {info['name']}{curr}")
            st.divider()
            # Progress bar
            done = sum(1 for i in wf["checklist"].values() if i["completed"])
            st.progress(done/7, text=f"Progress: {done}/7 stages")
            if wf["suggestion"]:
                st.caption(wf["suggestion"])

elif not query.strip() and submit:
    st.warning("Please enter a question.")

else:
    # Welcome state
    left, right = st.columns(2)
    with left:
        st.info(
            "👋 **Welcome to DS Mentor QA System!**\n\n"
            "This system uses **Hybrid BM25 + TF-IDF retrieval** "
            "(not ChatGPT-style generation) to give you grounded, "
            "verified data science answers.\n\n"
            "**What makes this different:**\n"
            "- Answers grounded in curated DS knowledge base\n"
            "- Detects if you're skipping important pipeline stages\n"
            "- Generates live charts for EDA questions\n"
            "- Tracks your workflow progress across the session"
        )
    with right:
        st.markdown("### 🗺️ Pipeline Stages")
        for s, name in STAGE_NAMES.items():
            st.markdown(
                f'<span class="stage-badge" style="background:{STAGE_COLORS[s]};margin:3px">'
                f'{STAGE_ICONS[s]} {s}. {name}</span>',
                unsafe_allow_html=True)

# ── Session history ────────────────────────────────────────────────────────
if st.session_state.get("history"):
    st.divider()
    hist = st.session_state["history"]
    with st.expander(f"📜 Session History ({len(hist)} queries)"):
        for i, h in enumerate(reversed(hist), 1):
            warn_badge = " ⚠️" if h.get("has_warning") else ""
            st.markdown(
                f"**{i}.** {h['query']} "
                f'<span class="stage-badge" style="background:{STAGE_COLORS[h["stage"]]}">'
                f'Stage {h["stage"]}: {h["stage_name"]}</span>{warn_badge}',
                unsafe_allow_html=True)
