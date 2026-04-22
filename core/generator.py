"""Template-based response generator with structured mentor output.
Produces clean, concise, professional responses without emojis or filler.
Sections: Answer / Why / Code / Watch Out / When to Use
Skill level calibrates explanation depth.
"""


class Generator:
    """Structured mentor response generator."""

    def __init__(self, config=None):
        self.config = config or {"provider": "template"}
        self.use_codet5 = bool(self.config.get("use_codet5", False))

    def generate(
        self,
        query: str,
        context: list,
        memory_context: str,
        difficulty: str,
        why_data: dict = None,
        pipeline_warnings: list = None,
        dataset_context: str = "",
        pedagogy_mode: str = "direct",
        planner_constraints: list = None,
        misconception_corrections: list = None,
        retry_feedback: str = "",
        stage_name: str = "",
    ) -> dict:
        """Build a structured mentor response from retrieved context.

        Returns dict with keys 'text' (formatted markdown) and 'code'.
        """
        extracted_code = ""
        context_text = "No direct context found for this question."

        if context:
            best_doc = context[0]
            if isinstance(best_doc, dict):
                extracted_code = best_doc.get("code", "")
                context_text = best_doc.get("answer", context_text)
            else:
                context_text = str(best_doc)

        # Optional neural code generation path (CodeT5).
        # If model is unavailable or output is invalid, keep retrieved code.
        if self.use_codet5:
            try:
                from models.finetune_codet5 import generate_codet5

                result = generate_codet5(query=query, stage_name=stage_name or "")
                candidate = str(result.get("code", "") or "")
                if candidate.strip() and bool(result.get("valid", False)):
                    extracted_code = candidate
            except Exception:
                pass

        skill = _skill_float(difficulty)
        answer_text = self._calibrate_depth(context_text, skill)

        formatted = self._format_response(
            answer=answer_text,
            code=extracted_code,
            why_data=why_data,
            skill_level=skill,
            warnings=pipeline_warnings,
            dataset_context=dataset_context,
            pedagogy_mode=pedagogy_mode,
            planner_constraints=planner_constraints or [],
            misconception_corrections=misconception_corrections or [],
            retry_feedback=retry_feedback,
        )

        return {"text": formatted, "code": extracted_code}

    @staticmethod
    def _calibrate_depth(text: str, skill_level: float) -> str:
        if skill_level < 0.3:
            return f"In simple terms: {text}"
        return text

    @staticmethod
    def _format_response(
        answer: str,
        code: str,
        why_data: dict = None,
        skill_level: float = 0.5,
        warnings: list = None,
        dataset_context: str = "",
        pedagogy_mode: str = "direct",
        planner_constraints: list = None,
        misconception_corrections: list = None,
        retry_feedback: str = "",
    ) -> str:
        """Assemble clean, professional markdown without emojis."""
        sections = []
        planner_constraints = planner_constraints or []
        misconception_corrections = misconception_corrections or []

        if dataset_context:
            sections.append(f"**Dataset context**\n{dataset_context}\n")

        if misconception_corrections:
            sections.append(f"**Correction before we proceed**\n{misconception_corrections[0]}")

        if pedagogy_mode != "direct":
            sections.append(f"**Teaching mode**\n{pedagogy_mode.replace('_', ' ').title()}")

        if retry_feedback:
            sections.append(f"**Quality note**\n{retry_feedback}")

        sections.append(f"**Answer**\n{answer}")

        if why_data and why_data.get("why"):
            sections.append(f"\n**Why this works**\n{why_data['why']}")

        if code:
            sections.append(f"\n**Code**\n```python\n{code}\n```")

        if why_data and why_data.get("common_pitfall"):
            sections.append(f"\n**Watch out**\n{why_data['common_pitfall']}")

        if why_data and why_data.get("when_to_use") and skill_level >= 0.3:
            sections.append(f"\n**When to use this**\n{why_data['when_to_use']}")

        if warnings:
            for w in warnings:
                sections.append(f"\n**Pipeline note** -- {w}")

        if planner_constraints:
            sections.append(f"\n**Plan constraints**\n- " + "\n- ".join(planner_constraints))

        if skill_level < 0.3:
            sections.append(
                "\n*Tip: focus on understanding what the code does "
                "before moving to the next stage.*"
            )
        elif skill_level > 0.7:
            sections.append(
                "\n*Consider edge cases and alternative approaches "
                "if your assumptions do not hold.*"
            )

        return "\n".join(sections)


def _skill_float(difficulty: str) -> float:
    return {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8}.get(difficulty, 0.5)
