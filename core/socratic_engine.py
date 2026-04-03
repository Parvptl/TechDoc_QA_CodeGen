"""
Socratic Mode: Sometimes guides the user to discover answers
instead of providing them directly.

Activates when:
1. Question is conceptual (not 'show me code')
2. User skill level is intermediate+ for this topic
3. Socratic mode hasn't been used in the last 3 interactions
4. User hasn't opted out ('just tell me')
"""
import re
from typing import Dict


class SocraticEngine:
    """Generates guiding questions instead of direct answers."""

    TEMPLATES: Dict[int, list] = {
        1: [
            "Before diving in, what do you think the most important factor for {topic} would be? What makes you think so?",
            "If you had to measure success for {topic} with a single number, what would you choose and why?",
        ],
        2: [
            "What format do you expect the data to be in? What issues might arise when loading it?",
            "If the dataset has millions of rows, how would that change your loading strategy?",
        ],
        3: [
            "Before exploring {topic}, what distribution would you expect? What would surprise you?",
            "If you found outliers in {topic}, what would that tell you about your data collection process?",
        ],
        4: [
            "What happens to your model if you skip preprocessing {topic}?",
            "If {topic} has missing values, what assumptions does each imputation method make about WHY the data is missing?",
        ],
        5: [
            "How do you decide which features to create? What makes a feature useful versus noisy?",
            "If you create too many features from {topic}, what problem might you introduce?",
        ],
        6: [
            "If your model gets 99% accuracy, is that good? What would make you suspicious?",
            "What is the risk of choosing a model based only on training accuracy?",
        ],
        7: [
            "Your model has high accuracy but low recall for the minority class. What does this mean in practical terms?",
            "If you could only look at one evaluation metric, which would you pick and why?",
        ],
    }

    HINT_TEMPLATES: Dict[int, str] = {
        1: "Think about what 'success' means in a business context and how you would measure it.",
        2: "Consider file size, encoding, and what happens when types are guessed incorrectly.",
        3: "Think about the relationship between the distribution shape and which methods are appropriate.",
        4: "Consider what information you are adding or losing with each transformation.",
        5: "Think about whether the new feature captures information that no existing feature already has.",
        6: "Consider the bias-variance tradeoff and what your learning curve looks like.",
        7: "Think about what each metric actually penalises and whether that matches your real-world cost.",
    }

    CONCEPTUAL_SIGNALS = frozenset([
        "why", "what happens", "explain", "difference between",
        "when should", "is it better", "what is the purpose",
        "what does", "why does", "how does", "what are the tradeoffs",
    ])
    CODE_SIGNALS = frozenset([
        "how to", "show me", "code for", "write", "implement",
        "example", "function", "snippet", "script",
    ])

    def __init__(self):
        self.interaction_counter: Dict[str, int] = {}
        self.socratic_cooldown: int = 3

    def should_activate(
        self,
        query: str,
        stage: int,
        skill_level: float,
        session_id: str = "default",
    ) -> bool:
        """
        Returns True when ALL conditions met:
        1. Query is conceptual
        2. Skill >= 0.3
        3. Cooldown elapsed
        4. Stage has templates
        5. User didn't say 'just tell me'
        """
        if self._is_bypass(query):
            return False
        if not self._is_conceptual_query(query):
            return False
        if skill_level < 0.3:
            return False
        if stage not in self.TEMPLATES:
            return False
        counter = self.interaction_counter.get(session_id, self.socratic_cooldown)
        if counter < self.socratic_cooldown:
            return False
        return True

    def generate_question(self, query: str, stage: int, retrieved_docs: list = None) -> dict:
        """
        Generate a Socratic guiding question.

        Returns:
            {
                'socratic_question': str,
                'hint': str,
                'full_answer_available': True,
            }
        """
        templates = self.TEMPLATES.get(stage, self.TEMPLATES[1])
        topic = self._extract_topic(query)

        idx = hash(query) % len(templates)
        question = templates[idx].replace("{topic}", topic)
        hint = self.HINT_TEMPLATES.get(stage, "Think step by step.")

        return {
            "socratic_question": question,
            "hint": hint,
            "full_answer_available": True,
        }

    def record_interaction(self, session_id: str, was_socratic: bool):
        """Track interaction count for cooldown logic."""
        if was_socratic:
            self.interaction_counter[session_id] = 0
        else:
            self.interaction_counter[session_id] = self.interaction_counter.get(session_id, 0) + 1

    def _is_conceptual_query(self, query: str) -> bool:
        """Returns True if query is conceptual, False if it's a code request."""
        q = query.lower()
        concept_score = sum(1 for sig in self.CONCEPTUAL_SIGNALS if sig in q)
        code_score = sum(1 for sig in self.CODE_SIGNALS if sig in q)
        return concept_score > code_score

    @staticmethod
    def _is_bypass(query: str) -> bool:
        """Detect if user wants to skip Socratic mode."""
        q = query.lower().strip()
        bypass_phrases = ["just tell me", "give me the answer", "skip", "don't ask"]
        return any(bp in q for bp in bypass_phrases)

    @staticmethod
    def _extract_topic(query: str) -> str:
        """Extract the main topic/noun from the query for template insertion."""
        q = query.strip()
        for prefix in ["why does", "what is", "explain", "what happens when",
                        "when should i", "is it better to", "how does"]:
            if q.lower().startswith(prefix):
                return q[len(prefix):].strip().rstrip("?").strip() or "this concept"
        words = re.findall(r"\b[a-zA-Z]{3,}\b", q)
        stopwords = {"the", "how", "what", "why", "does", "this", "that", "for", "and", "with"}
        content = [w for w in words if w.lower() not in stopwords]
        return " ".join(content[:4]) if content else "this concept"
