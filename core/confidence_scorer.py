class ConfidenceScorer:
    """Computes a composite reliability score for the generated response."""
    
    def __init__(self):
        pass

    def score(
        self,
        retrieval_score: float,
        classifier_confidence: float,
        generated_code_valid: bool,
        query_answer_overlap: float = 0.0,
        critic_min_axis: float = 1.0,
    ) -> float:
        """
        Retrieval score normalized relative to context ~[0-1]
        Classifier confidence [0-1]
        """
        parts = self.score_components(
            retrieval_score=retrieval_score,
            classifier_confidence=classifier_confidence,
            query_answer_overlap=query_answer_overlap,
            generated_code_valid=generated_code_valid,
            critic_min_axis=critic_min_axis,
        )
        return parts["score"]

    def score_components(
        self,
        retrieval_score: float,
        classifier_confidence: float,
        query_answer_overlap: float,
        generated_code_valid: bool,
        critic_min_axis: float = 1.0,
    ) -> dict:
        retrieval_factor = min(1.0, retrieval_score / 15.0) if retrieval_score > 0 else 0.0
        cls = max(0.0, min(1.0, classifier_confidence))
        overlap = max(0.0, min(1.0, query_answer_overlap))
        critic = max(0.0, min(1.0, critic_min_axis))
        code = 1.0 if generated_code_valid else 0.0

        score = (
            0.35 * retrieval_factor
            + 0.20 * cls
            + 0.20 * overlap
            + 0.20 * critic
            + 0.05 * code
        )
        score = max(0.0, min(1.0, score))
        return {
            "score": score,
            "retrieval_confidence": retrieval_factor,
            "classification_confidence": cls,
            "overlap_confidence": overlap,
            "critic_confidence": critic,
            "code_confidence": code,
            "label": self.label(score),
        }

    @staticmethod
    def label(score: float) -> str:
        if score >= 0.75:
            return "High confidence"
        if score >= 0.45:
            return "Moderate confidence - please verify"
        return "Low confidence - likely uncertain"
