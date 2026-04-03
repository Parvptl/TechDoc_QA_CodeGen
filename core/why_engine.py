"""
WHY Engine: Enriches every response with WHY and WHEN explanations.
Works in two modes:
- Dataset mode (default): Pulls pre-written WHY/WHEN from knowledge base
- LLM mode: Generates WHY/WHEN from context when dataset fields are empty
"""
import os


class WhyEngine:
    """
    Enriches responses with three layers:
    1. WHAT -- the direct answer (already provided by generator)
    2. WHY  -- mechanism/intuition behind the technique
    3. WHEN -- tradeoffs, alternatives, when to use what
    """

    def __init__(self, config: dict = None):
        """
        Config options:
        - mode: 'dataset' (use pre-written) or 'llm' (generate dynamically)
        - llm_provider: for LLM mode, which provider to use
        """
        self.config = config or {}
        self.mode = self.config.get("mode", "dataset")

    def enrich(self, response: dict, retrieved_docs: list, stage: int) -> dict:
        """
        Takes the generator's response and enriches it with WHY/WHEN layers.

        Returns dict with original keys plus:
          why, when_to_use, common_pitfall
        """
        enriched = dict(response)

        extracted = self._extract_from_docs(retrieved_docs)

        if self.mode == "llm" and not extracted.get("why"):
            extracted = self._generate_with_llm(response, retrieved_docs, stage)

        enriched["why"] = extracted.get("why", "")
        enriched["when_to_use"] = extracted.get("when_to_use", "")
        enriched["common_pitfall"] = extracted.get("common_pitfall", "")
        return enriched

    def _extract_from_docs(self, retrieved_docs: list) -> dict:
        """Pull WHY/WHEN/PITFALL from the top retrieved document's metadata."""
        result = {"why": "", "when_to_use": "", "common_pitfall": ""}
        if not retrieved_docs:
            return result

        top = retrieved_docs[0] if isinstance(retrieved_docs[0], dict) else {}
        result["why"] = top.get("why_explanation", "")
        result["when_to_use"] = top.get("when_to_use", "")
        result["common_pitfall"] = top.get("common_pitfall", "")
        return result

    def _generate_with_llm(self, response: dict, retrieved_docs: list, stage: int) -> dict:
        """
        Placeholder for LLM-powered WHY/WHEN generation.
        Falls back to empty strings when no LLM key is configured.
        """
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return {"why": "", "when_to_use": "", "common_pitfall": ""}

            import openai
            client = openai.OpenAI(api_key=api_key)
            context_text = response.get("text", "")[:500]
            prompt = (
                f"Given this data-science answer:\n{context_text}\n\n"
                "Provide:\n1. WHY this works (1-2 sentences, mechanism)\n"
                "2. WHEN to use this vs alternatives (1-2 sentences)\n"
                "3. Common pitfall (1 sentence)\n"
                "Format: WHY: ...\nWHEN: ...\nPITFALL: ..."
            )
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            text = resp.choices[0].message.content or ""
            return self._parse_llm_output(text)
        except Exception:
            return {"why": "", "when_to_use": "", "common_pitfall": ""}

    @staticmethod
    def _parse_llm_output(text: str) -> dict:
        """Parse structured LLM output into dict."""
        result = {"why": "", "when_to_use": "", "common_pitfall": ""}
        for line in text.split("\n"):
            lower = line.lower()
            if lower.startswith("why:"):
                result["why"] = line.split(":", 1)[1].strip()
            elif lower.startswith("when:"):
                result["when_to_use"] = line.split(":", 1)[1].strip()
            elif lower.startswith("pitfall:"):
                result["common_pitfall"] = line.split(":", 1)[1].strip()
        return result
