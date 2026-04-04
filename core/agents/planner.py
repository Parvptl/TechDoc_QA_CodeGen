"""Planner agent for lightweight task decomposition."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class SubTask:
    kind: str
    description: str
    priority: int = 1


@dataclass
class Plan:
    pedagogy_mode: str
    subtasks: List[SubTask] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)


class PlannerAgent:
    """Decomposes learner intent into small actionable subtasks."""

    _COMPLEXITY_MARKERS = (
        "and",
        "also",
        "then",
        "versus",
        "vs",
        "tradeoff",
        "trade-off",
        "pipeline",
        "end-to-end",
    )

    def plan(self, query: str, stage: int, skill_level: float, history: List[str]) -> Plan:
        q = (query or "").lower()
        subtasks: List[SubTask] = []
        key_terms: List[str] = []

        needs_code = any(token in q for token in ("code", "implement", "python", "sklearn", "pandas", "train"))
        asks_why = any(token in q for token in ("why", "intuition", "explain"))
        asks_compare = any(token in q for token in ("difference", "compare", "vs", "versus"))
        complex_query = (sum(marker in q for marker in self._COMPLEXITY_MARKERS) >= 2) or (len(q.split()) > 16)

        if asks_why:
            subtasks.append(SubTask(kind="concept", description="Explain concept and intuition", priority=1))
            key_terms.extend(["concept", "intuition", "why"])
        if asks_compare:
            subtasks.append(SubTask(kind="tradeoff", description="Compare alternatives and trade-offs", priority=1))
            key_terms.extend(["compare", "trade-off", "alternatives"])
        if needs_code:
            subtasks.append(SubTask(kind="code", description="Provide runnable code example", priority=2))
            key_terms.extend(["code", "example", "implementation"])

        subtasks.append(SubTask(kind="pitfall", description="Highlight one common pitfall", priority=2))
        key_terms.extend(["pitfall", "common mistake"])

        if complex_query:
            pedagogy_mode = "worked_example"
        elif skill_level < 0.35:
            pedagogy_mode = "guided"
        elif skill_level > 0.75:
            pedagogy_mode = "challenge"
        else:
            pedagogy_mode = "direct"

        constraints = []
        if stage in (6, 7):
            constraints.append("Include evaluation metric guidance")
        if history and len(history) > 2:
            constraints.append("Use session context and avoid repeating prior explanation verbatim")

        return Plan(
            pedagogy_mode=pedagogy_mode,
            subtasks=subtasks,
            constraints=constraints,
            key_terms=sorted(set(key_terms)),
        )
