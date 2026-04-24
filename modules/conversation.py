"""Multi-turn conversation manager with pronoun resolution, follow-up detection, and context injection."""
import re
from dataclasses import dataclass, field
from typing import Optional
import datetime


# ── Pronoun/reference patterns ────────────────────────────────────────────────
PRONOUN_PATTERNS = [
    (r"\bit\b",             "pronoun_it"),
    (r"\bthis\b",           "pronoun_this"),
    (r"\bthat\b",           "pronoun_that"),
    (r"\bthe model\b",      "ref_model"),
    (r"\bmy model\b",       "ref_model"),
    (r"\bthe data\b",       "ref_data"),
    (r"\bmy data\b",        "ref_data"),
    (r"\bthe dataset\b",    "ref_data"),
    (r"\bthe column\b",     "ref_column"),
    (r"\bthis feature\b",   "ref_column"),
    (r"\bthe result\b",     "ref_result"),
    (r"\bthe output\b",     "ref_result"),
    (r"\bthe same\b",       "ref_same"),
    (r"\bthat approach\b",  "ref_approach"),
    (r"\bthe above\b",      "ref_above"),
    (r"\bnow\b.*\?",        "followup_now"),
    (r"^how about",         "followup_how_about"),
    (r"^what about",        "followup_what_about"),
    (r"^and\b",             "followup_and"),
    (r"^also\b",            "followup_also"),
    (r"^but\b",             "followup_but"),
    (r"^why\b",             "followup_why"),
    (r"^can (i|you|we)\b",  "followup_can"),
]

FOLLOWUP_SIGNALS = {
    "pronoun_it", "pronoun_this", "pronoun_that",
    "ref_model", "ref_data", "ref_column", "ref_same",
    "followup_now", "followup_how_about", "followup_what_about",
    "followup_and", "followup_also", "followup_but", "followup_why",
}


@dataclass
class Turn:
    """A single conversation turn."""
    turn_id:    int
    query:      str
    stage_num:  int
    stage_name: str
    answer:     str          # explanation shown to user
    code:       str          # code shown to user
    timestamp:  str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    entities:   dict = field(default_factory=dict)  # columns, models mentioned


@dataclass
class ConversationSession:
    """Full multi-turn session state."""
    session_id:   str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    turns:        list = field(default_factory=list)
    entities:     dict = field(default_factory=lambda: {
        "columns":   [],
        "model":     None,
        "dataset":   None,
        "target":    None,
        "last_stage": None,
    })


class ConversationManager:
    """
    Multi-turn conversation manager for the DS Mentor QA system.

    Tracks conversation history, resolves references, and builds
    enriched queries that incorporate prior context.

    Usage:
        mgr = ConversationManager()
        enriched = mgr.process_turn("How do I handle missing values?")
        mgr.record_turn(query, stage_num, stage_name, answer, code)

        # Follow-up:
        enriched = mgr.process_turn("Now apply it to the Fare column")
        # → "Apply median imputation (from previous context) to the Fare column"
    """

    WINDOW = 5  # How many prior turns to keep in context

    def __init__(self):
        self.session = ConversationSession()

    # ── Main entry point ──────────────────────────────────────────────────
    def process_turn(self, raw_query: str) -> dict:
        """
        Process a new query, resolving any references to prior context.

        Returns:
            {
              "enriched_query":   str,   # query with resolved references
              "is_followup":      bool,
              "references":       list,  # detected reference types
              "injected_context": str,   # what context was injected
              "prior_stage":      int | None,
            }
        """
        references = self._detect_references(raw_query)
        is_followup = bool(references & FOLLOWUP_SIGNALS)

        if is_followup and self.session.turns:
            enriched, injected = self._resolve_references(raw_query, references)
        else:
            enriched = raw_query
            injected = ""

        # Extract entities from current query
        entities = self._extract_entities(raw_query)
        self._update_session_entities(entities)

        return {
            "enriched_query":   enriched,
            "is_followup":      is_followup,
            "references":       list(references),
            "injected_context": injected,
            "prior_stage":      self.session.entities.get("last_stage"),
        }

    def record_turn(self, query: str, stage_num: int, stage_name: str,
                    answer: str, code: str):
        """Record a completed turn into session history."""
        entities = self._extract_entities(query)
        turn = Turn(
            turn_id=len(self.session.turns) + 1,
            query=query, stage_num=stage_num, stage_name=stage_name,
            answer=answer, code=code, entities=entities,
        )
        self.session.turns.append(turn)
        self.session.entities["last_stage"] = stage_num

        # Update global entity memory
        self._update_session_entities(entities)

        # Keep window
        if len(self.session.turns) > self.WINDOW:
            self.session.turns = self.session.turns[-self.WINDOW:]

    # ── Reference detection ────────────────────────────────────────────────
    def _detect_references(self, query: str) -> set:
        """Detect which reference types are present in the query."""
        q = query.lower().strip()
        detected = set()
        for pattern, ref_type in PRONOUN_PATTERNS:
            if re.search(pattern, q):
                detected.add(ref_type)
        return detected

    # ── Reference resolution ───────────────────────────────────────────────
    def _resolve_references(self, query: str, references: set) -> tuple[str, str]:
        """
        Replace vague references with specific context from prior turns.
        Returns: (enriched_query, injected_context_description)
        """
        q = query
        injected_parts = []
        last = self.session.turns[-1] if self.session.turns else None

        # Resolve "it" / "this" / "that" → the topic of last turn
        if references & {"pronoun_it", "pronoun_this", "pronoun_that"}:
            if last:
                topic = last.stage_name
                # Try to be more specific
                if last.entities.get("columns"):
                    topic = f"the {last.entities['columns'][0]} column ({last.stage_name})"
                elif last.entities.get("model"):
                    topic = f"the {last.entities['model']} model"

                q = re.sub(r'\bit\b', topic, q, flags=re.I)
                q = re.sub(r'\bthis\b', topic, q, flags=re.I)
                q = re.sub(r'\bthat\b', topic, q, flags=re.I)
                injected_parts.append(f"'it/this/that' resolved to: {topic}")

        # Resolve "the model" → last known model type
        if "ref_model" in references and self.session.entities.get("model"):
            model = self.session.entities["model"]
            q = re.sub(r'\b(the model|my model)\b', f'the {model}', q, flags=re.I)
            injected_parts.append(f"model resolved to: {model}")

        # Resolve "the data" / "my data" → last known dataset
        if "ref_data" in references:
            dataset = self.session.entities.get("dataset") or "the training dataset"
            q = re.sub(r'\b(the data|my data|the dataset)\b', dataset, q, flags=re.I)
            injected_parts.append(f"data resolved to: {dataset}")

        # Resolve "the column" → last mentioned column
        if "ref_column" in references and self.session.entities.get("columns"):
            col = self.session.entities["columns"][-1]
            q = re.sub(r'\b(the column|this feature|this column)\b', f'the {col} column', q, flags=re.I)
            injected_parts.append(f"column resolved to: {col}")

        # Prepend prior stage context for follow-up signals
        if references & {"followup_now", "followup_how_about", "followup_what_about",
                         "followup_and", "followup_also"}:
            if last:
                context_prefix = (
                    f"[Context: Previously discussed {last.stage_name} — "
                    f"specifically: {last.query[:80]}] "
                )
                q = context_prefix + q
                injected_parts.append(f"added prior context from turn {last.turn_id}")

        return q, "; ".join(injected_parts)

    # ── Entity extraction ──────────────────────────────────────────────────
    def _extract_entities(self, query: str) -> dict:
        """Extract columns, models, datasets from a query."""
        entities = {"columns": [], "model": None, "dataset": None, "target": None}

        # Column names (quoted or known Titanic/common columns)
        entities["columns"] = re.findall(r"['\"]([A-Za-z_]\w*)['\"]", query)
        common_cols = ["Age", "Fare", "Sex", "Survived", "Pclass", "Embarked",
                       "SibSp", "Parch", "Name", "income", "target", "price", "churn"]
        for col in common_cols:
            if col.lower() in query.lower() and col not in entities["columns"]:
                entities["columns"].append(col)

        # Model type
        model_map = {
            "random forest": "Random Forest", "xgboost": "XGBoost",
            "lightgbm": "LightGBM", "logistic regression": "Logistic Regression",
            "svm": "SVM", "neural network": "Neural Network",
            "decision tree": "Decision Tree", "gradient boost": "Gradient Boosting",
        }
        q_lower = query.lower()
        for keyword, model_name in model_map.items():
            if keyword in q_lower:
                entities["model"] = model_name
                break

        # Dataset name
        ds_match = re.search(r"[\w./\\-]+\.(?:csv|xlsx|parquet|json)", query, re.I)
        if ds_match:
            entities["dataset"] = ds_match.group()

        # Target column
        target_match = re.search(r"(?:target|predict|label)\s*[=:]\s*['\"]?(\w+)['\"]?", query, re.I)
        if target_match:
            entities["target"] = target_match.group(1)

        return entities

    def _update_session_entities(self, entities: dict):
        """Merge new entities into session-level entity memory."""
        if entities.get("columns"):
            for col in entities["columns"]:
                if col not in self.session.entities["columns"]:
                    self.session.entities["columns"].append(col)
            # Keep last 10 mentioned columns
            self.session.entities["columns"] = self.session.entities["columns"][-10:]

        if entities.get("model"):
            self.session.entities["model"] = entities["model"]
        if entities.get("dataset"):
            self.session.entities["dataset"] = entities["dataset"]
        if entities.get("target"):
            self.session.entities["target"] = entities["target"]

    # ── Context summary ────────────────────────────────────────────────────
    def get_context_summary(self) -> dict:
        """Return current session context for display."""
        return {
            "turns":              len(self.session.turns),
            "known_columns":      self.session.entities["columns"],
            "known_model":        self.session.entities["model"],
            "known_dataset":      self.session.entities["dataset"],
            "known_target":       self.session.entities["target"],
            "last_stage":         self.session.entities["last_stage"],
            "recent_queries":     [t.query for t in self.session.turns[-3:]],
        }

    def get_conversation_history(self, n: int = 5) -> list[dict]:
        """Return last N turns formatted for display."""
        turns = self.session.turns[-n:]
        return [
            {
                "turn":       t.turn_id,
                "query":      t.query,
                "stage":      f"Stage {t.stage_num} — {t.stage_name}",
                "answer":     t.answer[:200] + "..." if len(t.answer) > 200 else t.answer,
                "timestamp":  t.timestamp,
            }
            for t in turns
        ]

    def reset(self):
        """Start a fresh conversation session."""
        self.session = ConversationSession()

    @property
    def turn_count(self) -> int:
        return len(self.session.turns)


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mgr = ConversationManager()

    print("=" * 60)
    print("MULTI-TURN CONVERSATION DEMO")
    print("=" * 60)

    conversation = [
        ("How do I load 'titanic.csv' with pandas?",
         2, "Data Loading",
         "Use pd.read_csv() to load your CSV file.",
         "import pandas as pd\ndf = pd.read_csv('titanic.csv')"),

        ("Now show me the distribution of the 'Age' column",
         3, "EDA",
         "Use seaborn histplot to visualize Age distribution.",
         "sns.histplot(df['Age'].dropna(), bins=30, kde=True)"),

        ("How do I fill missing values in it?",   # "it" = Age column
         4, "Preprocessing",
         "Use SimpleImputer with median strategy.",
         "df['Age'].fillna(df['Age'].median(), inplace=True)"),

        ("And what about the Fare column?",        # follow-up continuation
         4, "Preprocessing",
         "Same approach applies to Fare.",
         "df['Fare'].fillna(df['Fare'].median(), inplace=True)"),

        ("Now train a Random Forest on this data",  # "this data" = titanic
         6, "Modeling",
         "Train RandomForestClassifier on the preprocessed dataset.",
         "model = RandomForestClassifier(n_estimators=100)\nmodel.fit(X_train, y_train)"),

        ("How do I evaluate the model?",   # "the model" = Random Forest
         7, "Evaluation",
         "Use classification_report and roc_auc_score.",
         "print(classification_report(y_test, model.predict(X_test)))"),
    ]

    for i, (query, stage_num, stage_name, answer, code) in enumerate(conversation):
        print(f"\n{'─'*50}")
        print(f"Turn {i+1}: {query!r}")
        result = mgr.process_turn(query)
        print(f"  Is follow-up:  {result['is_followup']}")
        if result["references"]:
            print(f"  References:    {result['references']}")
        if result["injected_context"]:
            print(f"  Resolved:      {result['injected_context']}")
        if result["enriched_query"] != query:
            print(f"  Enriched:      {result['enriched_query'][:120]}")
        mgr.record_turn(query, stage_num, stage_name, answer, code)

    print(f"\n{'='*60}")
    print("SESSION CONTEXT SUMMARY")
    print("=" * 60)
    ctx = mgr.get_context_summary()
    for k, v in ctx.items():
        print(f"  {k}: {v}")
