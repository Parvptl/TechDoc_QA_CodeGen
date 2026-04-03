"""DS Mentor Pro -- Streamlit chat interface."""
import streamlit as st
import traceback
import sys
import os
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import MentorAgent

STAGE_NAMES = {
    1: "Problem Understanding",
    2: "Data Loading",
    3: "Exploratory Data Analysis",
    4: "Preprocessing",
    5: "Feature Engineering",
    6: "Modeling",
    7: "Evaluation",
}

st.set_page_config(page_title="DS Mentor Pro", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS -- clean light theme, neutral palette, no emojis
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #333;
        font-weight: 600;
    }

    /* Main header */
    .main-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 1.2rem;
    }

    /* Stage list in sidebar */
    .stage-item {
        display: flex;
        align-items: center;
        padding: 4px 0;
        font-size: 0.88rem;
        color: #444;
    }
    .stage-done {
        color: #2e7d32;
        font-weight: 600;
    }
    .stage-marker {
        display: inline-block;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        text-align: center;
        line-height: 18px;
        font-size: 0.72rem;
        margin-right: 8px;
        font-weight: 700;
    }
    .marker-done {
        background-color: #2e7d32;
        color: #fff;
    }
    .marker-pending {
        background-color: #e0e0e0;
        color: #777;
    }

    /* Dataset preview table */
    .dataset-summary {
        font-size: 0.85rem;
        color: #555;
        line-height: 1.5;
    }

    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .conf-high { background: #e8f5e9; color: #2e7d32; }
    .conf-mid  { background: #fff3e0; color: #e65100; }
    .conf-low  { background: #fce4ec; color: #c62828; }

    /* Pipeline warnings */
    .pipeline-note {
        background-color: #fff8e1;
        border-left: 3px solid #f9a825;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #5d4037;
        border-radius: 0 4px 4px 0;
    }
    .antipattern-note {
        background-color: #fce4ec;
        border-left: 3px solid #c62828;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #b71c1c;
        border-radius: 0 4px 4px 0;
    }

    /* Hide default streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Agent init (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_agent():
    docs = []
    try:
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "dataset.csv"
        )
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                docs = list(csv.DictReader(f))
                for r in docs:
                    r["stage"] = int(r.get("stage", 1))
    except Exception as e:
        print("Failed to load dataset:", e)
    return MentorAgent(data_docs=docs)


agent = load_agent()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "dataset_profile" not in st.session_state:
    st.session_state.dataset_profile = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p style="font-size:1.2rem;font-weight:700;color:#1a1a2e;">DS Mentor Pro</p>', unsafe_allow_html=True)
    st.caption("Agentic Data Science Tutor")
    st.divider()

    if st.button("Reset Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # -- Pipeline progress --
    st.markdown("### Pipeline Progress")
    completed = agent.tracker.completed_stages
    for s in range(1, 8):
        done = s in completed
        marker_cls = "marker-done" if done else "marker-pending"
        text_cls = "stage-done" if done else ""
        check = "&#10003;" if done else str(s)
        st.markdown(
            f'<div class="stage-item {text_cls}">'
            f'<span class="stage-marker {marker_cls}">{check}</span>'
            f'{STAGE_NAMES[s]}</div>',
            unsafe_allow_html=True,
        )
    done_count = len(completed)
    st.progress(done_count / 7, text=f"{done_count}/7 completed")

    # -- Next step suggestion --
    next_hint = agent.tracker.suggest_next_step()
    if next_hint:
        st.markdown(f'<div style="font-size:0.82rem;color:#555;margin-top:4px;">{next_hint}</div>', unsafe_allow_html=True)

    st.divider()

    # -- Skill levels --
    st.markdown("### Skill Levels")
    skills = agent.assessor.memory.profile.stage_skills
    for s in range(1, 8):
        val = skills.get(s, 0.5)
        label = "Beginner" if val < 0.3 else ("Intermediate" if val <= 0.7 else "Advanced")
        bar_pct = int(val * 100)
        st.markdown(
            f'<div style="font-size:0.84rem;color:#444;margin-bottom:2px;">'
            f'{STAGE_NAMES[s]}: <b>{label}</b></div>',
            unsafe_allow_html=True,
        )
        st.progress(val)

    st.divider()

    # -- Dataset upload --
    st.markdown("### Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        try:
            profile = agent.upload_dataset(file_bytes, uploaded_file.name)
            st.session_state.dataset_profile = profile
            st.success(f"Loaded: {profile['rows']} rows, {profile['columns']} cols")
        except Exception as exc:
            st.error(f"Could not parse file: {exc}")

    if st.session_state.dataset_profile:
        p = st.session_state.dataset_profile
        st.markdown(
            f'<div class="dataset-summary">'
            f'<b>{p["filename"]}</b><br>'
            f'{p["rows"]} rows &middot; {p["columns"]} columns<br>'
            f'Missing: {p["total_missing_pct"]}%'
            f'</div>',
            unsafe_allow_html=True,
        )
        if p.get("target_guess"):
            st.markdown(f'<div class="dataset-summary">Likely target: <b>{p["target_guess"]}</b></div>', unsafe_allow_html=True)

    st.divider()

    # -- Quick jump --
    st.markdown("### Quick Jump")
    for s, name in STAGE_NAMES.items():
        if st.button(name, key=f"jump_{s}", use_container_width=True):
            st.session_state.pending_question = f"Tell me about {name.lower()}"
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">DS Mentor Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Agentic Data Science Tutor</div>', unsafe_allow_html=True)

# -- Dataset preview (collapsible) --
if st.session_state.dataset_profile and agent.dataset_profiler.dataframe is not None:
    with st.expander("Dataset preview", expanded=False):
        st.dataframe(agent.dataset_profiler.dataframe.head(10), use_container_width=True)
        col_info = st.session_state.dataset_profile.get("column_info", [])
        if col_info:
            import pandas as pd
            info_df = pd.DataFrame([
                {
                    "Column": c["name"],
                    "Type": c["dtype"],
                    "Missing": f'{c["missing"]} ({c["missing_pct"]}%)',
                    "Unique": c["unique"],
                }
                for c in col_info
            ])
            st.dataframe(info_df, use_container_width=True, hide_index=True)


def _confidence_html(conf: float) -> str:
    if conf >= 70:
        cls = "conf-high"
    elif conf >= 40:
        cls = "conf-mid"
    else:
        cls = "conf-low"
    return f'<span class="confidence-badge {cls}">{conf:.0f}% confidence</span>'


# -- Render message history --
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("mode") == "socratic":
            st.markdown("**Think about this:**")
            st.markdown(msg["content"])
            if msg.get("hint"):
                with st.expander("Hint"):
                    st.markdown(msg["hint"])
        else:
            st.markdown(msg["content"])
            if msg.get("code"):
                st.code(msg["code"], language="python")

        if msg.get("confidence") and msg["role"] == "assistant":
            st.markdown(_confidence_html(msg["confidence"]), unsafe_allow_html=True)

        if msg.get("pipeline_warnings"):
            for w in msg["pipeline_warnings"]:
                st.markdown(f'<div class="pipeline-note">{w}</div>', unsafe_allow_html=True)
        if msg.get("antipattern_warnings"):
            for w in msg["antipattern_warnings"]:
                st.markdown(f'<div class="antipattern-note">{w}</div>', unsafe_allow_html=True)

        if msg.get("suggested_questions"):
            st.markdown("---")
            st.markdown("**Related questions:**")
            sq_cols = st.columns(min(len(msg["suggested_questions"]), 3))
            for i, sq in enumerate(msg["suggested_questions"]):
                label = sq["question"]
                if len(label) > 70:
                    label = label[:67] + "..."
                with sq_cols[i % len(sq_cols)]:
                    if st.button(label, key=f"follow_{msg_idx}_{i}"):
                        st.session_state.pending_question = sq["question"]
                        st.rerun()

# -- Chat input (must be top-level) --
auto_query = st.session_state.pending_question
if auto_query:
    st.session_state.pending_question = None
    prompt = auto_query
else:
    prompt = st.chat_input("Ask about your data science project...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            response = agent.process_sync(prompt)
            mode = response.get("mode", "direct")

            msg_data = {
                "role": "assistant",
                "content": response["text"],
                "code": response.get("code", ""),
                "mode": mode,
                "hint": response.get("hint", ""),
                "pipeline_warnings": response.get("pipeline_warnings", []),
                "antipattern_warnings": response.get("antipattern_warnings", []),
                "suggested_questions": response.get("suggested_questions", []),
                "stage_name": response["stage_name"],
                "confidence": response.get("confidence", 0),
            }

            with st.chat_message("assistant"):
                if mode == "socratic":
                    st.markdown("**Think about this:**")
                    st.markdown(response["text"])
                    if response.get("hint"):
                        with st.expander("Hint"):
                            st.markdown(response["hint"])
                else:
                    st.markdown(response["text"])
                    if response.get("code"):
                        st.code(response["code"], language="python")

                conf = response.get("confidence", 0)
                if conf:
                    st.markdown(_confidence_html(conf), unsafe_allow_html=True)

                if response.get("pipeline_warnings"):
                    for w in response["pipeline_warnings"]:
                        st.markdown(f'<div class="pipeline-note">{w}</div>', unsafe_allow_html=True)
                if response.get("antipattern_warnings"):
                    for w in response["antipattern_warnings"]:
                        st.markdown(f'<div class="antipattern-note">{w}</div>', unsafe_allow_html=True)

                if response.get("suggested_questions"):
                    st.markdown("---")
                    st.markdown("**Related questions:**")
                    sq_cols = st.columns(min(len(response["suggested_questions"]), 3))
                    for i, sq in enumerate(response["suggested_questions"]):
                        label = sq["question"]
                        if len(label) > 70:
                            label = label[:67] + "..."
                        with sq_cols[i % len(sq_cols)]:
                            if st.button(label, key=f"new_follow_{i}"):
                                st.session_state.pending_question = sq["question"]
                                st.rerun()

            st.session_state.messages.append(msg_data)

        except Exception as e:
            st.error(f"Error processing query: {e}")
            st.error(traceback.format_exc())
