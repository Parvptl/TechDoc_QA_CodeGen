"""Tracks completed DS pipeline stages, detects skips, and produces mentor guidance warnings."""

from dataclasses import dataclass, field
from typing import Optional
import datetime

# ── Stage metadata ────────────────────────────────────────────────────────────
STAGE_NAMES = {
    1: "Problem Understanding",
    2: "Data Loading",
    3: "Exploratory Data Analysis",
    4: "Preprocessing",
    5: "Feature Engineering",
    6: "Modeling",
    7: "Evaluation",
}

STAGE_IMPORTANCE = {
    1: "critical",
    2: "critical",
    3: "important",
    4: "critical",
    5: "important",
    6: "critical",
    7: "critical",
}

SKIP_WARNINGS = {
    3: ("⚠️  You skipped Exploratory Data Analysis.\n"
        "   Without EDA, you may miss data quality issues, outliers, or class imbalance.\n"
        "   Tip: Use df.describe(), sns.heatmap(df.corr()), and histograms first."),
    4: ("⚠️  You skipped Preprocessing.\n"
        "   Models trained on raw data with missing values or wrong dtypes will underperform.\n"
        "   Tip: Check df.isnull().sum() and handle NaN values before modeling."),
    5: ("⚠️  You skipped Feature Engineering.\n"
        "   Raw features are often suboptimal. Encoding and scaling can significantly boost accuracy.\n"
        "   Tip: Use LabelEncoder/OneHotEncoder for categoricals and StandardScaler for numerics."),
    2: ("⚠️  You skipped Data Loading.\n"
        "   Are you using data from a previous session? Make sure your data is properly loaded.\n"
        "   Tip: Always call df = pd.read_csv('data.csv') and check df.shape."),
}


@dataclass
class SessionHistory:
    """Tracks a single user session."""
    session_id:   str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    completed:    list[int]  = field(default_factory=list)   # stages done
    queries:      list[dict] = field(default_factory=list)   # full query log
    warnings_issued: list[str] = field(default_factory=list)


# ── WorkflowTracker ───────────────────────────────────────────────────────────
class WorkflowTracker:
    """
    Maintains stage history and issues mentor guidance.

    Usage:
        tracker = WorkflowTracker()
        result = tracker.process_query("How do I train an XGBoost model?")
        print(result["warning"])      # skip warning if any
        print(result["suggestion"])   # next recommended stage
    """

    def __init__(self):
        self.session = SessionHistory()

    # ── Core method ──────────────────────────────────────────────────────────
    def process_query(self, query: str, predicted_stage: Optional[int] = None) -> dict:
        """
        Given a user query (and optionally the pre-predicted stage),
        detect skips, log the event, and return guidance.

        Args:
            query:           Raw user question.
            predicted_stage: Stage number (1–7) from predict_stage(); if None,
                             will be inferred via keyword fallback.
        Returns:
            {
                "stage":       int,
                "stage_name":  str,
                "warning":     str | None,   # skip warning
                "suggestion":  str | None,   # what to do next
                "is_skip":     bool,
                "skipped":     list[int],    # which stages were skipped
                "checklist":   dict,         # full stage completion status
            }
        """
        if predicted_stage is None:
            predicted_stage = self._infer_stage(query)

        # Log query
        self.session.queries.append({
            "query": query,
            "stage": predicted_stage,
            "timestamp": datetime.datetime.now().isoformat(),
        })

        # Detect skipped stages
        skipped = self._detect_skips(predicted_stage)
        warning  = self._build_warning(skipped, predicted_stage) if skipped else None

        # Mark stage as completed
        if predicted_stage not in self.session.completed:
            self.session.completed.append(predicted_stage)
        self.session.completed.sort()

        # Suggestion: what's the next logical stage?
        suggestion = self._suggest_next()

        return {
            "stage":      predicted_stage,
            "stage_name": STAGE_NAMES[predicted_stage],
            "warning":    warning,
            "suggestion": suggestion,
            "is_skip":    bool(skipped),
            "skipped":    skipped,
            "checklist":  self.get_checklist(),
        }

    # ── Skip detection ────────────────────────────────────────────────────────
    def _detect_skips(self, current_stage: int) -> list[int]:
        """
        Return stages that were never visited but lie between the last
        completed stage and the current one.

        Example:
            completed = [2], current = 6  →  skipped = [3, 4, 5]
        """
        if not self.session.completed:
            # Allow starting at stage 1 silently; warn if jumping straight to modeling
            if current_stage > 2:
                return list(range(1, current_stage))
            return []

        last_done = max(self.session.completed)
        if current_stage <= last_done:
            return []   # revisiting an earlier stage — no skip

        gap = range(last_done + 1, current_stage)
        skipped = [s for s in gap if s not in self.session.completed]
        return skipped

    # ── Warning builder ───────────────────────────────────────────────────────
    def _build_warning(self, skipped: list[int], target: int) -> str:
        """Compose a human-readable warning about skipped stages."""
        skipped_names = [STAGE_NAMES[s] for s in skipped]
        names_str = ", ".join(skipped_names)
        target_name = STAGE_NAMES[target]

        header = (
            f"🚨 Mentor Warning: You jumped to **{target_name}** "
            f"without completing: {names_str}.\n"
        )
        details = []
        for s in skipped:
            if s in SKIP_WARNINGS:
                details.append(SKIP_WARNINGS[s])

        suggestion = "\nWould you like guidance on any of the skipped stages?"
        return header + "\n".join(details) + suggestion

    # ── Next-stage suggestion ─────────────────────────────────────────────────
    def _suggest_next(self) -> Optional[str]:
        """Recommend the next uncompleted stage in order."""
        for stage in range(1, 8):
            if stage not in self.session.completed:
                return (f"💡 Suggested next step: "
                        f"Stage {stage} — {STAGE_NAMES[stage]}")
        return "🎉 All pipeline stages completed! Ready for deployment."

    # ── Checklist ─────────────────────────────────────────────────────────────
    def get_checklist(self) -> dict:
        """Return completion status for all 7 stages."""
        return {
            stage: {
                "name":      name,
                "completed": stage in self.session.completed,
                "importance": STAGE_IMPORTANCE[stage],
            }
            for stage, name in STAGE_NAMES.items()
        }

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset_session(self):
        """Start a fresh session."""
        self.session = SessionHistory()

    def mark_complete(self, stage: int):
        """Manually mark a stage as completed."""
        if stage not in self.session.completed:
            self.session.completed.append(stage)
            self.session.completed.sort()

    # ── Keyword fallback ─────────────────────────────────────────────────────
    @staticmethod
    def _infer_stage(query: str) -> int:
        """Minimal keyword fallback (used when classifier is unavailable)."""
        q = query.lower()
        if any(k in q for k in ["problem", "goal", "objective", "target"]):       return 1
        if any(k in q for k in ["load", "read_csv", "import", "data"]):           return 2
        if any(k in q for k in ["plot", "visual", "distribut", "corr", "eda"]):   return 3
        if any(k in q for k in ["missing", "null", "clean", "preprocess"]):       return 4
        if any(k in q for k in ["feature", "encode", "scale", "engineer"]):       return 5
        if any(k in q for k in ["train", "model", "fit", "classifier"]):          return 6
        if any(k in q for k in ["evaluat", "metric", "accuracy", "score"]):       return 7
        return 1


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tracker = WorkflowTracker()

    print("=" * 60)
    print("SCENARIO 1: Normal workflow")
    print("=" * 60)
    for q in [
        "What is the goal of this project?",
        "How do I load a CSV file?",
        "Show me the distribution of the target variable",
        "How should I handle missing values?",
        "How do I encode the Sex column?",
        "How do I train a Random Forest?",
        "What is my model's accuracy?",
    ]:
        result = tracker.process_query(q)
        stage_str = f"Stage {result['stage']} — {result['stage_name']}"
        warn = f"\n  {result['warning']}" if result["warning"] else ""
        print(f"\nQ: {q!r}\n  → {stage_str}{warn}")

    print("\n" + "=" * 60)
    print("SCENARIO 2: Skip detection")
    print("=" * 60)
    tracker2 = WorkflowTracker()
    tracker2.session.completed = [2]  # Only data loading done

    result = tracker2.process_query("How do I train an XGBoost model?", predicted_stage=6)
    print(f"\nCompleted stages: {tracker2.session.completed[:-1]}")
    print(f"Query stage: Stage {result['stage']} — {result['stage_name']}")
    print(f"\n{result['warning']}")

    print("\n" + "=" * 60)
    print("SCENARIO 3: Checklist")
    print("=" * 60)
    for stage, info in result["checklist"].items():
        status = "✅" if info["completed"] else "⬜"
        print(f"  {status} Stage {stage}: {info['name']} [{info['importance']}]")
