"""
CONFIDENCE SCORER
=================
Scores answer reliability based on:
  1. Retrieval score from the RAG system
  2. Stage classifier confidence
  3. Query-answer semantic overlap (TF-IDF cosine)
  4. Code syntax validity

Returns a 0–1 confidence score + human-readable label.
"""
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
_fitted = False

def _fit_vectorizer(texts):
    global _fitted
    _vec.fit(texts)
    _fitted = True

def _query_doc_sim(query: str, doc: str) -> float:
    """TF-IDF cosine similarity between query and retrieved doc."""
    global _fitted
    if not _fitted:
        _fit_vectorizer([query, doc])
    try:
        vecs = _vec.transform([query, doc])
        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        return float(sim)
    except Exception:
        return 0.5

def _code_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def _retrieval_score_to_prob(score: float) -> float:
    """Convert raw RRF retrieval score to 0–1 probability."""
    # Typical RRF scores range from 0.015 to 0.055
    # Map to 0.5–1.0 range
    normalized = min(1.0, max(0.0, (score - 0.01) / 0.05))
    return 0.5 + 0.5 * normalized

def score_answer(
    query: str,
    retrieved_explanation: str,
    retrieved_code: str,
    retrieval_score: float,
    classifier_confidence: float,
) -> dict:
    """
    Compute a composite confidence score for an answer.

    Args:
        query:                  User's question
        retrieved_explanation:  Retrieved explanation text
        retrieved_code:         Retrieved code snippet
        retrieval_score:        Raw RRF retrieval score
        classifier_confidence:  Stage classifier probability

    Returns:
        {
            "overall": float (0–1),
            "label":   str ("High" | "Medium" | "Low"),
            "breakdown": dict,
        }
    """
    components = {}

    # 1. Retrieval quality
    components["retrieval"] = _retrieval_score_to_prob(retrieval_score)

    # 2. Stage classifier confidence (already 0–1)
    components["classifier"] = min(1.0, classifier_confidence)

    # 3. Query-answer semantic overlap
    combined_doc = retrieved_explanation + " " + retrieved_code
    components["semantic_overlap"] = _query_doc_sim(query, combined_doc)

    # 4. Code validity bonus
    components["code_valid"] = 1.0 if _code_syntax_valid(retrieved_code) else 0.3

    # Weighted composite
    weights = {"retrieval": 0.35, "classifier": 0.25,
               "semantic_overlap": 0.30, "code_valid": 0.10}
    overall = sum(components[k] * weights[k] for k in weights)

    # Label
    if overall >= 0.72:
        label = "High"
        color = "#2ecc71"
        icon  = "🟢"
    elif overall >= 0.50:
        label = "Medium"
        color = "#f39c12"
        icon  = "🟡"
    else:
        label = "Low"
        color = "#e74c3c"
        icon  = "🔴"

    return {
        "overall":   round(overall, 4),
        "label":     label,
        "color":     color,
        "icon":      icon,
        "breakdown": {k: round(v, 4) for k, v in components.items()},
    }


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = score_answer(
        query="How do I fill missing values in Age with median?",
        retrieved_explanation="How do I handle missing values in a numeric column?",
        retrieved_code="from sklearn.impute import SimpleImputer\nimputer = SimpleImputer(strategy='median')\ndf['Age'] = imputer.fit_transform(df[['Age']])",
        retrieval_score=0.0452,
        classifier_confidence=0.672,
    )
    print("Confidence Score:", result["overall"])
    print("Label:", result["icon"], result["label"])
    print("Breakdown:", result["breakdown"])

    # Low confidence example
    result2 = score_answer(
        query="What is the meaning of life?",
        retrieved_explanation="How do I train a Random Forest classifier?",
        retrieved_code="model = RandomForestClassifier()",
        retrieval_score=0.012,
        classifier_confidence=0.15,
    )
    print("\nLow confidence example:", result2["label"], result2["overall"])
