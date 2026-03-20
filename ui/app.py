"""
DS MENTOR QA SYSTEM — COMPLETE APP
Member 1: 700+ dataset | Member 2: Hybrid RAG | Member 3: Code Gen | Member 4: Conversation + Workflow + Viz
"""
import sys, base64
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="DS Mentor QA", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

@st.cache_resource(show_spinner="Loading retrieval index…")
def load_retriever():
    from modules.retrieval import DSRetriever
    return DSRetriever(str(ROOT / "data/dataset.csv"))

@st.cache_resource(show_spinner="Loading stage classifier…")
def load_classifier():
    from models.stage_classifier import predict_stage_with_confidence, extract_stage_num
    return predict_stage_with_confidence, extract_stage_num

STAGE_NAMES  = {1:"Problem Understanding",2:"Data Loading",3:"Exploratory Data Analysis",
                4:"Preprocessing",5:"Feature Engineering",6:"Modeling",7:"Evaluation"}
STAGE_COLORS = {1:"#6c5ce7",2:"#0984e3",3:"#00b894",4:"#fdcb6e",5:"#e17055",6:"#d63031",7:"#2d3436"}
STAGE_ICONS  = {1:"🎯",2:"📂",3:"📊",4:"🧹",5:"⚙️",6:"🤖",7:"📈"}

st.markdown("""<style>
.stage-badge{display:inline-block;padding:4px 14px;border-radius:20px;color:white;font-weight:700;font-size:.85em;margin-bottom:8px}
.panel-title{font-size:.82em;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:#636e72;margin-bottom:4px}
.warn-box{background:#fff3cd;border-left:5px solid #ffc107;padding:10px 14px;border-radius:4px;margin:6px 0}
.ok-box{background:#d4edda;border-left:5px solid #28a745;padding:8px 14px;border-radius:4px;margin:6px 0}
.info-box{background:#e8f4fd;border-left:5px solid #0984e3;padding:8px 14px;border-radius:4px;margin:6px 0}
.rcard{background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:8px 12px;margin:3px 0;font-size:.88em}
.cpill{display:inline-block;background:#e0e7ff;color:#3730a3;padding:2px 8px;border-radius:12px;font-size:.78em;margin:2px}
</style>""", unsafe_allow_html=True)

if "workflow" not in st.session_state:
    from modules.workflow import WorkflowTracker
    st.session_state.workflow = WorkflowTracker()
if "conversation" not in st.session_state:
    from modules.conversation import ConversationManager
    st.session_state.conversation = ConversationManager()
if "history" not in st.session_state:
    st.session_state.history = []

workflow     = st.session_state.workflow
conversation = st.session_state.conversation

with st.sidebar:
    st.title("🎓 DS Mentor")
    st.caption("BM25+TF-IDF · Code Gen · Multi-turn · Workflow")
    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.clear(); st.rerun()
    st.markdown("### ⚡ Quick Jump")
    for s, name in STAGE_NAMES.items():
        if st.button(f"{STAGE_ICONS[s]} {s}. {name}", key=f"j{s}", use_container_width=True):
            st.session_state.forced_stage = s
            st.session_state.preset_query = f"Tell me about {name.lower()}"
            st.rerun()
    st.divider()
    ctx_sum = conversation.get_context_summary()
    if ctx_sum["turns"] > 0:
        st.markdown("### 🧠 Session Memory")
        if ctx_sum["known_model"]:
            st.markdown(f'<span class="cpill">🤖 {ctx_sum["known_model"]}</span>', unsafe_allow_html=True)
        if ctx_sum["known_dataset"]:
            st.markdown(f'<span class="cpill">📂 {ctx_sum["known_dataset"]}</span>', unsafe_allow_html=True)
        for col in ctx_sum["known_columns"][-4:]:
            st.markdown(f'<span class="cpill">📋 {col}</span>', unsafe_allow_html=True)
        st.caption(f"{ctx_sum['turns']} turn(s) in memory")
    if st.session_state.history:
        st.divider()
        h = st.session_state.history
        c1,c2 = st.columns(2)
        c1.metric("Queries", len(h))
        c2.metric("Stages", f"{len(set(x['stage'] for x in h))}/7")

st.title("🎓 Data Science Mentor QA System")
st.caption("Hybrid BM25+TF-IDF retrieval · Context-aware code generation · Multi-turn conversation · Pipeline workflow guidance")

with st.expander("🔧 System Status", expanded=False):
    try:
        _r = load_retriever()
        s  = _r.get_stats()
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("📚 Dataset", s["total_documents"])
        c2.metric("🔍 BM25",   "✅" if s["bm25_available"] else "⚠️")
        c3.metric("📐 Vocab",  f"{s['tfidf_vocab_size']:,}")
        c4.metric("🧠 CodeGen","✅")
        c5.metric("💬 Conv",   "✅")
        st.success("✅ All modules ready.")
    except Exception as e:
        st.error(f"Error: {e}")
st.divider()

cq, cb = st.columns([5,1])
with cq:
    preset = st.session_state.pop("preset_query", "")
    query  = st.text_input("Question:", value=preset,
                           placeholder="e.g. How do I fill missing values in the 'Age' column?",
                           label_visibility="collapsed")
with cb:
    submit = st.button("Ask 🔍", use_container_width=True, type="primary")

with st.expander("💡 Examples — click to try"):
    exs = [
        "What is the goal of a Titanic survival prediction task?",
        "How do I load 'train.csv' with pandas?",
        "Plot distribution of the 'Age' column",
        "How do I fill missing values in 'Fare' using median?",
        "Encode the 'Sex' column",
        "Train a Random Forest with 200 estimators",
        "Compute AUC and plot confusion matrix",
        "How do I fill missing values in it?",
        "Now evaluate the model",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(exs):
        if cols[i%3].button(ex, key=f"e{i}", use_container_width=True):
            st.session_state.preset_query = ex; st.rerun()
st.divider()

if submit and query.strip():
    retriever = load_retriever()
    predict_fn, extract_num = load_classifier()

    conv_result     = conversation.process_turn(query)
    effective_query = conv_result["enriched_query"]

    forced = st.session_state.pop("forced_stage", None)
    if forced:
        stage_num, confidence = forced, 1.0
    else:
        stage_label, confidence = predict_fn(effective_query)
        stage_num = extract_num(stage_label)

    stage_name = STAGE_NAMES.get(stage_num, "Unknown")
    color      = STAGE_COLORS.get(stage_num, "#999")
    wf         = workflow.process_query(effective_query, predicted_stage=stage_num)
    results    = retriever.retrieve(effective_query, top_k=5, predicted_stage=stage_num)
    top_doc    = results[0] if results else None

    from modules.code_generator import generate_code, validate_code_syntax
    code_result = generate_code(effective_query, stage_num, top_doc["code"] if top_doc else None)
    final_code  = code_result["code"]
    code_valid  = validate_code_syntax(final_code)

    explanation = top_doc["explanation"] if top_doc else "No matching document found."
    conversation.record_turn(query, stage_num, stage_name, explanation, final_code)
    st.session_state.history.append({
        "query": query, "stage": stage_num, "stage_name": stage_name,
        "has_warning": bool(wf["warning"]), "confidence": confidence,
        "is_followup": conv_result["is_followup"],
    })

    b1, b2, b3 = st.columns([3,1,1])
    with b1:
        st.markdown(f'<span class="stage-badge" style="background:{color}">'
                    f'{STAGE_ICONS.get(stage_num,"")} Stage {stage_num} — {stage_name}</span>',
                    unsafe_allow_html=True)
    b2.metric("Confidence", f"{confidence:.0%}")
    b3.metric("Code method", code_result["method"].title())

    if conv_result["is_followup"]:
        st.markdown(f'<div class="info-box">💬 <b>Follow-up detected</b> — '
                    f'{conv_result["injected_context"] or "prior context applied"}</div>',
                    unsafe_allow_html=True)
    if wf["warning"]:
        st.markdown(f'<div class="warn-box">{wf["warning"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ok-box">✅ Good workflow progression!</div>', unsafe_allow_html=True)

    tl, tr = st.columns(2)
    bl, br = st.columns(2)

    with tl:
        st.markdown('<p class="panel-title">💬 Explanation (RAG Retrieved)</p>', unsafe_allow_html=True)
        with st.container(border=True):
            if top_doc:
                st.markdown(f"**{top_doc['explanation']}**")
                st.caption(f"Source: {top_doc.get('source','curated')} · "
                           f"Difficulty: {top_doc.get('difficulty','intermediate')} · "
                           f"Score: {top_doc['retrieval_score']:.5f}")
            else:
                st.info("No matching document found. Try rephrasing.")
            if len(results) > 1:
                with st.expander(f"📑 {len(results)-1} more results"):
                    for r in results[1:]:
                        m = "✓" if int(r["pipeline_stage"])==stage_num else "○"
                        st.markdown(f'<div class="rcard"><b>{m} [S{r["pipeline_stage"]}]</b> '
                                    f'{r["explanation"][:85]}… <small>{r["retrieval_score"]:.5f}</small></div>',
                                    unsafe_allow_html=True)

    with tr:
        st.markdown('<p class="panel-title">💻 Generated Code</p>', unsafe_allow_html=True)
        with st.container(border=True):
            if not code_valid["valid"]:
                st.warning(f"⚠️ {code_valid['error']}")
            st.code(final_code, language="python")
            cctx = code_result.get("context", {})
            tags = []
            if cctx.get("columns"): tags.append(f"cols={cctx['columns']}")
            if cctx.get("model"):   tags.append(f"model={cctx['model']}")
            if cctx.get("metric"):  tags.append(f"metric={cctx['metric']}")
            if cctx.get("n"):       tags.append(f"n={cctx['n']}")
            if tags: st.caption("Extracted context: " + " · ".join(tags))

    with bl:
        st.markdown('<p class="panel-title">📊 Visualization</p>', unsafe_allow_html=True)
        with st.container(border=True):
            if stage_num == 3:
                with st.spinner("Generating plot…"):
                    from modules.visualization import handle_eda_query
                    viz = handle_eda_query(effective_query)
                if viz["success"] and viz.get("image_b64"):
                    st.image(base64.b64decode(viz["image_b64"]), use_container_width=True)
                    with st.expander("🔍 Viz code"):
                        st.code(viz["generated_code"], language="python")
                else:
                    st.warning(f"Viz error: {str(viz.get('error',''))[:200]}")
            else:
                st.info(f"📊 Live plots for Stage 3 (EDA) queries.\n"
                        f"Current: Stage {stage_num} — {stage_name}\n"
                        f"Try: 'Show distribution of the Age column'")

    with br:
        st.markdown('<p class="panel-title">📋 Workflow Checklist</p>', unsafe_allow_html=True)
        with st.container(border=True):
            for s, info in wf["checklist"].items():
                icon = "✅" if info["completed"] else "⬜"
                imp  = "🔴" if info["importance"]=="critical" else "🟡"
                curr = " ← **now**" if s==stage_num else ""
                st.markdown(f"{icon} {imp} **{s}.** {info['name']}{curr}")
            st.divider()
            done = sum(1 for i in wf["checklist"].values() if i["completed"])
            st.progress(done/7, text=f"Progress: {done}/7 stages")
            if wf["suggestion"]: st.caption(wf["suggestion"])

elif submit:
    st.warning("Please enter a question.")

else:
    l, r = st.columns(2)
    with l:
        st.info("👋 **Welcome to DS Mentor QA System!**\n\n"
                "Four novel modules:\n\n"
                "🔍 **BM25+TF-IDF RAG** — 700-example curated dataset, RRF fusion, stage re-ranking\n\n"
                "🧠 **Context-aware Code Gen** — adapts to your column names, model, metric\n\n"
                "💬 **Multi-turn Conversation** — resolves 'it', 'the model', 'this data'\n\n"
                "📋 **Pipeline Guidance** — warns when you skip stages, tracks progress")
    with r:
        st.markdown("### 🗺️ 7-Stage DS Pipeline")
        for s, name in STAGE_NAMES.items():
            st.markdown(f'<span class="stage-badge" style="background:{STAGE_COLORS[s]};margin:3px">'
                        f'{STAGE_ICONS[s]} {s}. {name}</span>', unsafe_allow_html=True)
        st.markdown("\n### 💡 Multi-turn demo:")
        for step in ["1. *'How do I load train.csv?'*",
                     "2. *'Show distribution of Age'*",
                     "3. *'Fill missing values in it'* ← resolves 'it'",
                     "4. *'Train a Random Forest'*",
                     "5. *'Now evaluate the model'* ← resolves 'the model'"]:
            st.markdown(step)

if st.session_state.get("history"):
    st.divider()
    hist = st.session_state["history"]
    with st.expander(f"📜 Session History — {len(hist)} queries"):
        for i, h in enumerate(reversed(hist), 1):
            tags = ("  💬" if h.get("is_followup") else "") + ("  ⚠️" if h.get("has_warning") else "")
            st.markdown(
                f"**{i}.** {h['query'][:80]} "
                f'<span class="stage-badge" style="background:{STAGE_COLORS[h["stage"]]}">'
                f'S{h["stage"]}: {h["stage_name"]}</span>{tags}',
                unsafe_allow_html=True)
