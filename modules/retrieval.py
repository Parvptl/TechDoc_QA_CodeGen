"""
PART 2 — RAG RETRIEVAL ENGINE
==============================
Hybrid retrieval: BM25 (keyword) + TF-IDF cosine (semantic) with re-ranking.
This is the core novel contribution vs ChatGPT — retrieves grounded answers
from the curated DS dataset rather than hallucinating.

Architecture:
  Query → [BM25 score] + [TF-IDF cosine score] → RRF fusion → Top-K results → Re-rank by stage

Usage:
    from modules.retrieval import DSRetriever
    retriever = DSRetriever("data/dataset.csv")
    results = retriever.retrieve("How do I handle missing values?", top_k=5)
"""
import csv
import re
import math
from pathlib import Path
from typing import Optional

# ── BM25 implementation (no external deps beyond rank_bm25) ─────────────────
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("[WARN] rank_bm25 not installed. Run: pip install rank_bm25")

# ── TF-IDF cosine (sklearn always available) ─────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ── Text preprocessing ───────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s_]", " ", text)
    return text.split()


def build_corpus_text(row: dict) -> str:
    """Combine explanation + code into one searchable string."""
    explanation = row.get("explanation", "")
    code = row.get("code", "")
    stage_name = row.get("stage_name", "")
    return f"{explanation} {stage_name} {code}"


# ── Reciprocal Rank Fusion ───────────────────────────────────────────────────
def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """
    Combine multiple ranked lists into one via RRF.
    Higher score = more relevant.
    """
    scores: dict[int, float] = {}
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Main Retriever ────────────────────────────────────────────────────────────
class DSRetriever:
    """
    Hybrid BM25 + TF-IDF retriever for the DS Mentor QA system.

    Novel contributions vs ChatGPT:
    1. Answers are grounded in curated, verified DS knowledge base
    2. Hybrid retrieval reduces vocabulary mismatch (BM25) and catches
       semantic similarity (TF-IDF cosine)
    3. Stage-aware re-ranking boosts results matching the classified stage
    """

    def __init__(self, dataset_path: str = "data/dataset.csv"):
        self.dataset_path = dataset_path
        self.documents: list[dict] = []
        self.corpus_texts: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: Optional[object] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._loaded = False
        self._load_and_index()

    def _load_and_index(self):
        """Load dataset and build BM25 + TF-IDF indexes."""
        path = Path(self.dataset_path)
        if not path.exists():
            print(f"[WARN] Dataset not found at {path}. Call build_dataset() first.")
            return

        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # Add stage name for richer indexing
        STAGE_NAMES = {1:"Problem Understanding",2:"Data Loading",3:"Exploratory Data Analysis",
                       4:"Preprocessing",5:"Feature Engineering",6:"Modeling",7:"Evaluation"}
        for row in rows:
            row["stage_name"] = STAGE_NAMES.get(int(row.get("pipeline_stage", 1)), "")
        self.documents = rows

        # Build corpus
        self.corpus_texts = [build_corpus_text(r) for r in rows]
        self.tokenized_corpus = [tokenize(t) for t in self.corpus_texts]

        # BM25 index
        if BM25_AVAILABLE:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            print(f"[INFO] BM25 index built: {len(rows)} documents")
        else:
            print("[INFO] BM25 unavailable — using TF-IDF only")

        # TF-IDF index
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
            min_df=1,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_texts)
        print(f"[INFO] TF-IDF index built: {self.tfidf_matrix.shape}")
        self._loaded = True

    def _bm25_retrieve(self, query: str, top_k: int = 20) -> list[int]:
        """Return top-k document indices by BM25 score."""
        if not BM25_AVAILABLE or self.bm25 is None:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k]
        return ranked.tolist()

    def _tfidf_retrieve(self, query: str, top_k: int = 20) -> list[int]:
        """Return top-k document indices by TF-IDF cosine similarity."""
        query_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked = np.argsort(sims)[::-1][:top_k]
        return ranked.tolist()

    def _stage_boost(self, doc_idx: int, predicted_stage: Optional[int],
                     base_score: float) -> float:
        """Boost score if document matches the predicted stage."""
        if predicted_stage is None:
            return base_score
        doc_stage = int(self.documents[doc_idx].get("pipeline_stage", 0))
        if doc_stage == predicted_stage:
            return base_score * 1.4   # 40% boost for stage match
        elif abs(doc_stage - predicted_stage) == 1:
            return base_score * 1.1   # 10% boost for adjacent stage
        return base_score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        predicted_stage: Optional[int] = None,
        stage_filter: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve top-k most relevant documents for a query.

        Args:
            query:           User's natural language question.
            top_k:           Number of results to return.
            predicted_stage: Stage number for score boosting.
            stage_filter:    If set, only return docs from this stage.

        Returns:
            List of dicts with keys: explanation, code, pipeline_stage,
            stage_name, source, difficulty, retrieval_score, rank.
        """
        if not self._loaded or not self.documents:
            return []

        # Get ranked lists from both retrievers
        bm25_ranked  = self._bm25_retrieve(query, top_k=30)
        tfidf_ranked = self._tfidf_retrieve(query, top_k=30)

        # Fuse via RRF
        all_rankings = [r for r in [bm25_ranked, tfidf_ranked] if r]
        if not all_rankings:
            return []

        fused = reciprocal_rank_fusion(all_rankings)

        # Apply stage boost and filter
        boosted = []
        for doc_idx, score in fused:
            if stage_filter is not None:
                doc_stage = int(self.documents[doc_idx].get("pipeline_stage", 0))
                if doc_stage != stage_filter:
                    continue
            boosted_score = self._stage_boost(doc_idx, predicted_stage, score)
            boosted.append((doc_idx, boosted_score))

        boosted.sort(key=lambda x: x[1], reverse=True)

        # Build result objects
        results = []
        for rank, (doc_idx, score) in enumerate(boosted[:top_k], 1):
            doc = dict(self.documents[doc_idx])
            doc["retrieval_score"] = round(score, 6)
            doc["rank"] = rank
            results.append(doc)

        return results

    def get_stage_examples(self, stage: int, n: int = 3) -> list[dict]:
        """Return n random examples for a given stage."""
        import random
        stage_docs = [d for d in self.documents
                      if int(d.get("pipeline_stage", 0)) == stage]
        return random.sample(stage_docs, min(n, len(stage_docs)))

    def get_stats(self) -> dict:
        """Return index statistics."""
        if not self._loaded:
            return {}
        stage_counts = {}
        for doc in self.documents:
            s = int(doc.get("pipeline_stage", 0))
            stage_counts[s] = stage_counts.get(s, 0) + 1
        return {
            "total_documents": len(self.documents),
            "stage_counts": stage_counts,
            "bm25_available": BM25_AVAILABLE,
            "tfidf_vocab_size": len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else 0,
        }

    def evaluate_retrieval(self, test_queries: list[dict]) -> dict:
        """
        Compute MRR and Recall@K on a test set.

        Args:
            test_queries: list of {query, relevant_stage} dicts.

        Returns:
            {mrr, recall_at_1, recall_at_3, recall_at_5}
        """
        reciprocal_ranks = []
        recall_at = {1: [], 3: [], 5: []}

        for item in test_queries:
            query = item["query"]
            relevant_stage = item["relevant_stage"]
            results = self.retrieve(query, top_k=5)

            # Find rank of first relevant result
            rr = 0.0
            for rank_idx, r in enumerate(results, 1):
                if int(r["pipeline_stage"]) == relevant_stage:
                    rr = 1.0 / rank_idx
                    break
            reciprocal_ranks.append(rr)

            # Recall@K: is there at least one correct result in top-K?
            for k in [1, 3, 5]:
                top_k_stages = [int(r["pipeline_stage"]) for r in results[:k]]
                recall_at[k].append(1.0 if relevant_stage in top_k_stages else 0.0)

        return {
            "mrr":         round(np.mean(reciprocal_ranks), 4),
            "recall_at_1": round(np.mean(recall_at[1]), 4),
            "recall_at_3": round(np.mean(recall_at[3]), 4),
            "recall_at_5": round(np.mean(recall_at[5]), 4),
            "n_queries":   len(test_queries),
        }


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = DSRetriever("data/dataset.csv")
    print("\n=== Retriever Stats ===")
    print(retriever.get_stats())

    queries = [
        ("How do I fill missing values in Age?", 4),
        ("Plot a correlation heatmap", 3),
        ("Train a Random Forest model", 6),
        ("What is the AUC score?", 7),
        ("How do I encode categorical features?", 5),
    ]

    print("\n=== Retrieval Demo ===")
    for q, expected_stage in queries:
        results = retriever.retrieve(q, top_k=3, predicted_stage=expected_stage)
        print(f"\nQuery: {q!r}  (expected stage {expected_stage})")
        for r in results:
            stage_match = "✓" if int(r["pipeline_stage"]) == expected_stage else "✗"
            print(f"  {stage_match} [Stage {r['pipeline_stage']}] {r['explanation'][:60]}... (score={r['retrieval_score']:.5f})")

    print("\n=== Retrieval Evaluation ===")
    test_set = [{"query": q, "relevant_stage": s} for q, s in queries]
    metrics = retriever.evaluate_retrieval(test_set)
    print(f"MRR:        {metrics['mrr']:.4f}")
    print(f"Recall@1:   {metrics['recall_at_1']:.4f}")
    print(f"Recall@3:   {metrics['recall_at_3']:.4f}")
    print(f"Recall@5:   {metrics['recall_at_5']:.4f}")
