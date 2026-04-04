from fastapi.testclient import TestClient

from services.api import app


client = TestClient(app)


def test_v1_chat_route():
    res = client.post(
        "/api/v1/chat",
        json={"query": "How do I load csv data?", "session_id": "test_v1"},
    )
    assert res.status_code == 200
    body = res.json()
    assert "text" in body
    assert "stage_num" in body


def test_v1_session_progress_route():
    client.post("/api/v1/chat", json={"query": "How to evaluate a model?", "session_id": "progress_v1"})
    res = client.get("/api/v1/session/progress_v1/progress")
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == "progress_v1"
    assert "stage_mastery" in body
    assert "questions_asked" in body
    assert "quiz_count" in body
    assert "checkpoint_total" in body


def test_v1_session_report_route():
    client.post("/api/v1/chat", json={"query": "How to load csv?", "session_id": "report_v1"})
    res = client.get("/api/v1/session/report_v1/report?format=markdown")
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == "report_v1"
    assert body["format"] == "markdown"
    assert "path" in body
    assert "# DS Mentor Pro - Session Learning Report" in body.get("content", "")


def test_v1_session_dashboard_route():
    client.post("/api/v1/chat", json={"query": "How to preprocess data?", "session_id": "dash_v1"})
    res = client.get("/api/v1/session/dash_v1/dashboard")
    assert res.status_code == 200
    body = res.json()
    assert "summary" in body
    assert "trends" in body


def test_v1_quiz_generation_route():
    res = client.post("/api/v1/quiz/generate", json={"session_id": "quiz_v1"})
    assert res.status_code == 200
    body = res.json()
    assert "questions" in body
    assert len(body["questions"]) >= 1
    assert "id" in body["questions"][0]


def test_v1_project_checkpoint_route():
    # First upload minimal dataset to enable project mode in shared agent.
    files = {"file": ("mini.csv", b"target,x\n1,10\n0,20\n", "text/csv")}
    client.post("/api/v1/upload?session_id=checkpoint_v1", files=files)

    reject = client.post(
        "/api/v1/project/checkpoint",
        json={"session_id": "checkpoint_v1", "stage_num": 1, "evidence": "short"},
    )
    assert reject.status_code == 200
    reject_body = reject.json()
    assert reject_body["accepted"] is False
    next_stage = int(reject_body.get("next_required_stage", 1))
    universal_evidence = (
        "Defined objective target metric baseline; used read_csv with shape dtype columns; "
        "ran plot distribution correlation outlier missing checks; applied impute fillna encode scale preprocess; "
        "performed feature selection pca engineer transform; trained model fit with cross_val random forest; "
        "evaluated using f1 auc precision recall confusion."
    )

    accept = client.post(
        "/api/v1/project/checkpoint",
        json={
            "session_id": "checkpoint_v1",
            "stage_num": next_stage,
            "evidence": universal_evidence,
        },
    )
    assert accept.status_code == 200
    assert accept.json()["accepted"] is True
    assert "assessment" in accept.json()


def test_v1_quiz_grade_route():
    gen = client.post("/api/v1/quiz/generate", json={"session_id": "grade_v1"}).json()
    questions = gen["questions"]
    answers = []
    for q in questions:
        if q["id"] == "q1":
            answers.append({"id": "q1", "answer": "Precision/Recall/F1"})
        elif q["id"] == "q2":
            answers.append({"id": "q2", "answer": "train_test_split"})
        elif q["id"] == "q3":
            answers.append({"id": "q3", "answer": "A major pitfall is leakage from test set into training."})

    res = client.post("/api/v1/quiz/grade", json={"session_id": "grade_v1", "answers": answers})
    assert res.status_code == 200
    body = res.json()
    assert body["score"] >= 0.66
    assert body["mastery_updated"] is True


def test_v1_notebook_review_includes_antipatterns():
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# EDA"]},
            {
                "cell_type": "code",
                "source": [
                    "from sklearn.preprocessing import StandardScaler\n",
                    "scaler = StandardScaler()\n",
                    "X = scaler.fit_transform(X)\n",
                ],
            },
        ]
    }
    res = client.post("/api/v1/notebook/review", json={"notebook_json": notebook})
    assert res.status_code == 200
    body = res.json()
    assert "antipattern_warnings" in body
    assert isinstance(body["antipattern_warnings"], list)
