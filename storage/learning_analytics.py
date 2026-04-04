"""SQLite-backed learning analytics for DS Mentor Pro."""
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any


class LearningAnalyticsStore:
    """Stores query-level telemetry and aggregates session progress."""

    def __init__(self, db_path: str = "storage/learning_analytics.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quiz_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    stage_num INTEGER NOT NULL,
                    score REAL NOT NULL,
                    correct INTEGER NOT NULL,
                    total INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    stage_num INTEGER NOT NULL,
                    accepted INTEGER NOT NULL,
                    score REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    stage_num INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    confidence_label TEXT NOT NULL,
                    misconceptions_count INTEGER NOT NULL,
                    antipattern_count INTEGER NOT NULL,
                    response_time_ms INTEGER NOT NULL,
                    mode TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def log_event(
        self,
        session_id: str,
        stage_num: int,
        confidence: float,
        confidence_label: str,
        misconceptions_count: int,
        antipattern_count: int,
        response_time_ms: int,
        mode: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_events
                (ts, session_id, stage_num, confidence, confidence_label,
                 misconceptions_count, antipattern_count, response_time_ms, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    session_id,
                    int(stage_num),
                    float(confidence),
                    str(confidence_label),
                    int(misconceptions_count),
                    int(antipattern_count),
                    int(response_time_ms),
                    str(mode),
                ),
            )
            conn.commit()

    def session_summary(self, session_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as query_count,
                    AVG(confidence) as avg_confidence,
                    AVG(response_time_ms) as avg_response_ms,
                    SUM(misconceptions_count) as misconceptions_total,
                    SUM(antipattern_count) as antipattern_total
                FROM query_events
                WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*),
                    AVG(score)
                FROM quiz_events
                WHERE session_id = ?
                """,
                (session_id,),
            )
            quiz_row = cursor.fetchone()
            cursor = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END),
                    COUNT(*)
                FROM checkpoint_events
                WHERE session_id = ?
                """,
                (session_id,),
            )
            checkpoint_row = cursor.fetchone()
            cursor = conn.execute(
                """
                SELECT stage_num, COUNT(*)
                FROM query_events
                WHERE session_id = ?
                GROUP BY stage_num
                ORDER BY stage_num
                """,
                (session_id,),
            )
            by_stage = {int(stage): int(count) for stage, count in cursor.fetchall()}

        return {
            "query_count": int(row[0] or 0),
            "avg_confidence": round(float(row[1] or 0.0), 2),
            "avg_response_ms": int(round(float(row[2] or 0.0))),
            "misconceptions_total": int(row[3] or 0),
            "antipattern_total": int(row[4] or 0),
            "quiz_count": int(quiz_row[0] or 0),
            "avg_quiz_score": round(float(quiz_row[1] or 0.0), 3),
            "checkpoint_passed": int(checkpoint_row[0] or 0),
            "checkpoint_total": int(checkpoint_row[1] or 0),
            "stage_query_distribution": by_stage,
        }

    def log_quiz_event(self, session_id: str, stage_num: int, score: float, correct: int, total: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO quiz_events (ts, session_id, stage_num, score, correct, total)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    session_id,
                    int(stage_num),
                    float(score),
                    int(correct),
                    int(total),
                ),
            )
            conn.commit()

    def log_checkpoint_event(self, session_id: str, stage_num: int, accepted: bool, score: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoint_events (ts, session_id, stage_num, accepted, score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    session_id,
                    int(stage_num),
                    1 if accepted else 0,
                    float(score),
                ),
            )
            conn.commit()

    def session_trends(self, session_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            q = conn.execute(
                """
                SELECT id, confidence, response_time_ms, misconceptions_count
                FROM query_events
                WHERE session_id = ?
                ORDER BY id
                """,
                (session_id,),
            ).fetchall()
            quiz = conn.execute(
                """
                SELECT id, score
                FROM quiz_events
                WHERE session_id = ?
                ORDER BY id
                """,
                (session_id,),
            ).fetchall()
            chk = conn.execute(
                """
                SELECT id, accepted, score
                FROM checkpoint_events
                WHERE session_id = ?
                ORDER BY id
                """,
                (session_id,),
            ).fetchall()

        return {
            "confidence_series": [{"step": int(r[0]), "value": float(r[1])} for r in q],
            "latency_series": [{"step": int(r[0]), "value": int(r[2])} for r in q],
            "misconception_series": [{"step": int(r[0]), "value": int(r[3])} for r in q],
            "quiz_series": [{"step": int(r[0]), "value": float(r[1])} for r in quiz],
            "checkpoint_series": [{"step": int(r[0]), "accepted": int(r[1]), "score": float(r[2])} for r in chk],
        }
