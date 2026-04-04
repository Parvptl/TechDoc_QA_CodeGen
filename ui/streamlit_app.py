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
if "quiz_payload" not in st.session_state:
    st.session_state.quiz_payload = None
if "quiz_result" not in st.session_state:
    st.session_state.quiz_result = None
if "checkpoint_feedback" not in st.session_state:
    st.session_state.checkpoint_feedback = ""
if "report_export" not in st.session_state:
    st.session_state.report_export = None

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

    # -- Learning analytics snapshot --
    st.markdown("### Learning Analytics")
    progress_snapshot = agent.get_progress("default")
    c1, c2 = st.columns(2)
    c1.metric("Questions", progress_snapshot.get("questions_asked", 0))
    c2.metric("Misconceptions", progress_snapshot.get("misconceptions", 0))
    c3, c4 = st.columns(2)
    c3.metric("Avg confidence", f"{progress_snapshot.get('avg_confidence', 0):.1f}%")
    c4.metric("Avg latency", f"{progress_snapshot.get('avg_response_ms', 0)} ms")
    c5, c6 = st.columns(2)
    c5.metric("Quizzes", progress_snapshot.get("quiz_count", 0))
    c6.metric("Avg quiz", f"{int(progress_snapshot.get('avg_quiz_score', 0) * 100)}%")
    c7, c8 = st.columns(2)
    c7.metric("Checkpoints passed", progress_snapshot.get("checkpoint_passed", 0))
    c8.metric("Checkpoints total", progress_snapshot.get("checkpoint_total", 0))
    with st.expander("Stage query distribution", expanded=False):
        stage_dist = progress_snapshot.get("stage_distribution", {})
        if stage_dist:
            for stage_num in sorted(stage_dist.keys()):
                st.write(f"Stage {stage_num}: {stage_dist[stage_num]} queries")
        else:
            st.caption("No analytics yet for this session.")

    with st.expander("Learning Dashboard", expanded=False):
        dashboard = agent.get_learning_dashboard("default")
        trends = dashboard.get("trends", {})
        if trends.get("confidence_series"):
            st.caption("Confidence trend")
            st.line_chart(
                {
                    "confidence": [x["value"] for x in trends["confidence_series"]],
                }
            )
        if trends.get("quiz_series"):
            st.caption("Quiz score trend")
            st.line_chart(
                {
                    "quiz_score": [x["value"] for x in trends["quiz_series"]],
                }
            )
        if trends.get("latency_series"):
            st.caption("Response latency trend (ms)")
            st.line_chart(
                {
                    "latency_ms": [x["value"] for x in trends["latency_series"]],
                }
            )
        cexp1, cexp2 = st.columns(2)
        if cexp1.button("Export report (.md)", key="export_report_md"):
            st.session_state.report_export = agent.export_learning_report("default", "markdown")
        if cexp2.button("Export report (.json)", key="export_report_json"):
            st.session_state.report_export = agent.export_learning_report("default", "json")
        if st.session_state.report_export:
            rep = st.session_state.report_export
            st.caption(f"Saved report: {rep.get('path', '')}")
            preview = rep.get("content", "")
            if len(preview) > 1200:
                preview = preview[:1200] + "\n..."
            st.text_area("Report preview", value=preview, height=180)

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
    project_plan = agent.get_project_plan()
    with st.expander("Project-Based Learning Mode", expanded=True):
        st.markdown(f"**{project_plan.get('title', 'Project plan')}**")
        st.caption(project_plan.get("brief", ""))
        for cp in project_plan.get("checkpoints", []):
            status = "Focus" if cp.get("status") == "focus" else "Ready"
            lock = "Unlocked" if cp.get("unlocked", False) else "Locked"
            st.write(
                f"Stage {cp['stage_num']} - {cp['stage_name']} "
                f"({int(cp.get('mastery', 0) * 100)}% mastery) [{status} | {lock}]"
            )
            st.caption(cp.get("checkpoint", ""))
        next_stage = int(project_plan.get("next_required_stage", 1))
        st.markdown(f"**Submit checkpoint for Stage {next_stage}**")
        checkpoint_evidence = st.text_area(
            "Checkpoint evidence (brief explanation or code snippet)",
            key="checkpoint_evidence_text",
            height=110,
        )
        if st.button(f"Submit Stage {next_stage} checkpoint", key="submit_checkpoint_btn"):
            result = agent.submit_project_checkpoint("default", next_stage, checkpoint_evidence)
            st.session_state.checkpoint_feedback = result.get("message", "")
            if result.get("accepted"):
                st.success(st.session_state.checkpoint_feedback)
                st.rerun()
            else:
                st.warning(st.session_state.checkpoint_feedback)
                assess = result.get("assessment", {})
                if assess:
                    st.caption(
                        f"Score: {assess.get('score', 0)} (min {assess.get('min_required_score', 0)}) | "
                        f"Matched keywords: {', '.join(assess.get('matched_keywords', [])) or 'none'}"
                    )
        if st.session_state.checkpoint_feedback:
            st.caption(st.session_state.checkpoint_feedback)

    with st.expander("Quiz Mode", expanded=False):
        if st.button("Generate quiz", key="generate_quiz_btn"):
            st.session_state.quiz_payload = agent.generate_quiz("default")
            st.session_state.quiz_result = None
            st.rerun()

        quiz_payload = st.session_state.quiz_payload
        if quiz_payload:
            st.caption(
                f"Stage {quiz_payload.get('stage')} quiz | difficulty: {quiz_payload.get('difficulty', 'intermediate')}"
            )
            answers = []
            for q in quiz_payload.get("questions", []):
                qid = q.get("id", "")
                st.markdown(f"**{qid}. {q.get('question', '')}**")
                if q.get("type") == "mcq":
                    selected = st.radio(
                        f"Choose answer for {qid}",
                        options=q.get("options", []),
                        key=f"quiz_ans_{qid}",
                        label_visibility="collapsed",
                    )
                    answers.append({"id": qid, "answer": selected})
                else:
                    text_ans = st.text_input(
                        f"Answer for {qid}",
                        key=f"quiz_ans_{qid}",
                        label_visibility="collapsed",
                    )
                    answers.append({"id": qid, "answer": text_ans})

            if st.button("Submit quiz", key="submit_quiz_btn"):
                st.session_state.quiz_result = agent.grade_quiz(
                    session_id="default",
                    stage=quiz_payload.get("stage"),
                    answers=answers,
                )
                st.rerun()

            if st.session_state.quiz_result:
                qr = st.session_state.quiz_result
                st.success(f"Quiz score: {int(qr.get('score', 0) * 100)}% ({qr.get('correct', 0)}/{qr.get('total', 0)})")

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
            if msg.get("code_explained"):
                with st.expander("Line-by-line explanation"):
                    st.code(msg["code_explained"], language="python")

        if msg.get("confidence") and msg["role"] == "assistant":
            st.markdown(_confidence_html(msg["confidence"]), unsafe_allow_html=True)

        if msg.get("pipeline_warnings"):
            for w in msg["pipeline_warnings"]:
                st.markdown(f'<div class="pipeline-note">{w}</div>', unsafe_allow_html=True)
        if msg.get("antipattern_warnings"):
            for w in msg["antipattern_warnings"]:
                st.markdown(f'<div class="antipattern-note">{w}</div>', unsafe_allow_html=True)
        if msg.get("misconception_alerts"):
            for m in msg["misconception_alerts"]:
                st.markdown(
                    f'<div class="pipeline-note"><b>Misconception check:</b> {m.get("correction", "")}</div>',
                    unsafe_allow_html=True,
                )
        if msg.get("confidence_label") and msg["role"] == "assistant":
            st.caption(msg["confidence_label"])
        if msg.get("critic_feedback") and msg["role"] == "assistant":
            with st.expander("Quality review"):
                st.markdown(msg["critic_feedback"])

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
                "code_explained": response.get("code_explained", ""),
                "mode": mode,
                "hint": response.get("hint", ""),
                "pipeline_warnings": response.get("pipeline_warnings", []),
                "antipattern_warnings": response.get("antipattern_warnings", []),
                "suggested_questions": response.get("suggested_questions", []),
                "stage_name": response["stage_name"],
                "confidence": response.get("confidence", 0),
                "confidence_label": response.get("confidence_label", ""),
                "misconception_alerts": response.get("misconception_alerts", []),
                "critic_feedback": response.get("critic", {}).get("feedback", ""),
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
                    if response.get("code_explained"):
                        with st.expander("Line-by-line explanation"):
                            st.code(response["code_explained"], language="python")

                conf = response.get("confidence", 0)
                if conf:
                    st.markdown(_confidence_html(conf), unsafe_allow_html=True)

                if response.get("pipeline_warnings"):
                    for w in response["pipeline_warnings"]:
                        st.markdown(f'<div class="pipeline-note">{w}</div>', unsafe_allow_html=True)
                if response.get("antipattern_warnings"):
                    for w in response["antipattern_warnings"]:
                        st.markdown(f'<div class="antipattern-note">{w}</div>', unsafe_allow_html=True)
                if response.get("misconception_alerts"):
                    for m in response["misconception_alerts"]:
                        st.markdown(
                            f'<div class="pipeline-note"><b>Misconception check:</b> {m.get("correction", "")}</div>',
                            unsafe_allow_html=True,
                        )
                if response.get("confidence_label"):
                    st.caption(response["confidence_label"])
                critic_feedback = response.get("critic", {}).get("feedback", "")
                if critic_feedback:
                    with st.expander("Quality review"):
                        st.markdown(critic_feedback)

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
