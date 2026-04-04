from core.agent import MentorAgent
from core.agents.planner import PlannerAgent
from core.agents.critic import CriticAgent
from core.detection.misconception import MisconceptionDetector
from core.retrieval.query_expand import QueryExpander
from core.debug_assistant import DebugAssistant
from core.code_annotator import CodeAnnotator
from core.checkpoint_assessor import CheckpointAssessor
from core.quiz_engine import QuizEngine


def test_query_expansion_adds_related_terms():
    expander = QueryExpander()
    out = expander.expand("How do I clean data?")
    assert "preprocessing" in out.lower() or "data wrangling" in out.lower()


def test_misconception_detector_finds_accuracy_imbalance():
    detector = MisconceptionDetector(min_hits=1)
    hits = detector.detect("I got 99% accuracy on an imbalanced dataset. Is this perfect?", stage=7)
    assert isinstance(hits, list)
    assert len(hits) >= 1


def test_planner_creates_subtasks():
    planner = PlannerAgent()
    plan = planner.plan(
        query="How do I handle class imbalance and show code?",
        stage=6,
        skill_level=0.5,
        history=["How to train random forest?"],
    )
    assert len(plan.subtasks) >= 2
    assert plan.pedagogy_mode in {"direct", "guided", "challenge", "worked_example"}


def test_critic_retry_gate():
    critic = CriticAgent(retry_threshold=0.6)
    score = critic.evaluate(
        response_text="Generic answer unrelated.",
        retrieved_context=[{"answer": "Use stratified split and F1 score for imbalanced data."}],
        query="How to evaluate imbalanced data?",
        difficulty="beginner",
        required_subtasks=["concept", "pitfall"],
    )
    assert isinstance(score.feedback, str)
    assert critic.should_retry(score) in {True, False}


def test_debug_assistant_traceback_parsing():
    debug = DebugAssistant()
    suggestion = debug.suggest(
        "Traceback (most recent call last):\n"
        "  File 'x.py', line 3, in <module>\n"
        "ValueError: could not convert string to float: 'male'"
    )
    assert suggestion.error_type == "ValueError"
    assert suggestion.category == "encoding"
    assert suggestion.line_number == 3


def test_agent_returns_new_expansion_fields():
    agent = MentorAgent()
    res = agent.process_sync("How do I handle class imbalance in random forest?")
    assert "critic" in res
    assert "plan_subtasks" in res
    assert "pedagogy_mode" in res
    assert "confidence_label" in res
    assert "misconception_alerts" in res


def test_project_mode_plan_after_upload():
    agent = MentorAgent()
    csv_bytes = b"target,age,fare\n1,22,7.25\n0,38,71.83\n"
    agent.upload_dataset(csv_bytes, "tiny.csv")
    plan = agent.get_project_plan()
    assert plan["mode"] == "project"
    assert len(plan.get("checkpoints", [])) == 7


def test_progress_endpoint_data_shape():
    agent = MentorAgent()
    sid = "progress_shape_test"
    agent.process_sync("How do I load csv?", session_id=sid)
    progress = agent.get_progress(sid)
    assert progress["session_id"] == sid
    assert "stage_mastery" in progress
    assert "questions_asked" in progress


def test_code_annotator_adds_inline_explanations():
    ann = CodeAnnotator()
    code = "from sklearn.model_selection import train_test_split\nX_train, X_test = train_test_split(X)"
    out = ann.annotate(code)
    assert "Split data into train and test partitions" in out


def test_project_mode_gates_stage_jump_after_upload():
    agent = MentorAgent()
    csv_bytes = b"target,age,fare\n1,22,7.25\n0,38,71.83\n"
    agent.upload_dataset(csv_bytes, "tiny.csv")
    res = agent.process_sync("Train a random forest model now", session_id="gate_test")
    assert res["mode"] == "project_gate"
    assert "checkpoint" in res["text"].lower()


def test_checkpoint_assessor_scores_stage_keywords():
    assessor = CheckpointAssessor(min_required_score=0.55)
    result = assessor.assess(
        1,
        "We define the objective, target variable, baseline metric, and expected business goal.",
    )
    assert result.score > 0.55
    assert result.passed is True


def test_quiz_engine_grading():
    engine = QuizEngine()
    questions = engine.generate("Evaluation")
    answers = [
        {"id": "q1", "answer": "Precision/Recall/F1"},
        {"id": "q2", "answer": "train_test_split"},
        {"id": "q3", "answer": "A common pitfall is leakage."},
    ]
    grade = engine.grade(questions, answers)
    assert grade.total == 3
    assert grade.score >= 0.66


def test_agent_quiz_methods_and_mastery_update():
    agent = MentorAgent()
    quiz = agent.generate_quiz(session_id="agent_quiz")
    assert "questions" in quiz and len(quiz["questions"]) >= 1
    answers = []
    for q in quiz["questions"]:
        if q["id"] == "q1":
            answers.append({"id": "q1", "answer": "Precision/Recall/F1"})
        elif q["id"] == "q2":
            answers.append({"id": "q2", "answer": "train_test_split"})
        else:
            answers.append({"id": q["id"], "answer": "A pitfall is leakage from test data."})
    result = agent.grade_quiz("agent_quiz", answers=answers, stage=quiz["stage"])
    assert result["mastery_updated"] is True
    assert result["score"] >= 0.66


def test_agent_submit_project_checkpoint():
    agent = MentorAgent()
    sid = "agent_checkpoint"
    csv_bytes = b"target,age,fare\n1,22,7.25\n0,38,71.83\n"
    agent.upload_dataset(csv_bytes, "tiny.csv")
    rejected = agent.submit_project_checkpoint(sid, stage_num=1, evidence="short")
    assert rejected["accepted"] is False
    next_stage = int(rejected.get("next_required_stage", 1))
    accepted = agent.submit_project_checkpoint(
        sid,
        stage_num=next_stage,
        evidence=(
            "Defined objective target metric baseline; used read_csv with shape dtype columns; "
            "ran plot distribution correlation outlier missing checks; applied impute fillna encode scale preprocess; "
            "performed feature selection pca engineer transform; trained model fit with cross_val random forest; "
            "evaluated using f1 auc precision recall confusion."
        ),
    )
    assert accepted["accepted"] is True


def test_agent_learning_dashboard_contains_trends():
    agent = MentorAgent()
    sid = "dashboard_test"
    agent.process_sync("How do I load csv?", session_id=sid)
    quiz = agent.generate_quiz(session_id=sid)
    answers = []
    for q in quiz["questions"]:
        if q["id"] == "q1":
            answers.append({"id": q["id"], "answer": "Precision/Recall/F1"})
        elif q["id"] == "q2":
            answers.append({"id": q["id"], "answer": "train_test_split"})
        else:
            answers.append({"id": q["id"], "answer": "A common pitfall is leakage."})
    agent.grade_quiz(session_id=sid, answers=answers, stage=quiz["stage"])
    agent.submit_project_checkpoint(
        session_id=sid,
        stage_num=1,
        evidence=(
            "Defined objective target metric baseline; used read_csv with shape dtype columns; "
            "ran plot distribution correlation outlier missing checks; applied impute fillna encode scale preprocess; "
            "performed feature selection pca engineer transform; trained model fit with cross_val random forest; "
            "evaluated using f1 auc precision recall confusion."
        ),
    )
    dashboard = agent.get_learning_dashboard(session_id=sid)
    assert "summary" in dashboard
    assert "trends" in dashboard
    assert "confidence_series" in dashboard["trends"]


def test_agent_export_learning_report():
    agent = MentorAgent()
    sid = "export_report_test"
    agent.process_sync("How do I read csv?", session_id=sid)
    md = agent.export_learning_report(session_id=sid, fmt="markdown")
    js = agent.export_learning_report(session_id=sid, fmt="json")
    assert md["format"] == "markdown"
    assert js["format"] == "json"
    assert "Session Learning Report" in md["content"]
    assert '"summary"' in js["content"]
