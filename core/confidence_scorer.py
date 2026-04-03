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
    ) -> float:
        """
        Retrieval score normalized relative to context ~[0-1]
        Classifier confidence [0-1]
        """
        # Base limits
        score = 0.25
        
        # Retrieval quality factor
        retrieval_factor = min(1.0, retrieval_score / 15.0) if retrieval_score > 0 else 0
        score += 0.30 * retrieval_factor
        
        # Classifier certainty factor
        score += 0.20 * max(0.0, min(1.0, classifier_confidence))

        # Query-answer lexical overlap factor
        score += 0.20 * max(0.0, min(1.0, query_answer_overlap))
        
        # Code generation penalty/bonus
        if generated_code_valid:
            score += 0.05
        else:
            score -= 0.15
            
        return max(0.0, min(1.0, score))
