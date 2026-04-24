"""Hybrid retriever: BM25 + TF-IDF + optional FAISS fused via RRF with stage-aware re-ranking."""

import csv
import re
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

# ── BM25 ──────────────────────────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# ── TF-IDF ────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── FAISS (optional dense retrieval) ─────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ── Sentence embeddings (optional) ───────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

STAGE_NAMES = {
    1: "Problem Understanding",  2: "Data Loading",
    3: "Exploratory Data Analysis", 4: "Preprocessing",
    5: "Feature Engineering",    6: "Modeling",
    7: "Evaluation",
}


# ── Text helpers ───────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s_]", " ", text.lower()).split()


def build_doc_text(row: dict) -> str:
    stage_num = int(row.get("pipeline_stage", row.get("stage", 1)))
    return (
        f"{row.get('query', '')} "
        f"{row.get('answer', '')} "
        f"{row.get('why_explanation', '')} "
        f"{STAGE_NAMES.get(stage_num, '')} "
        f"{row.get('code', '')}"
    )


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────
def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── FAISS dense index ──────────────────────────────────────────────────────────
class FAISSIndex:
    """
    Dense FAISS index using Sentence-BERT embeddings.
    Falls back gracefully if faiss or sentence_transformers are unavailable.
    """
    INDEX_PATH  = "rag/faiss_index.bin"
    EMBEDS_PATH = "rag/doc_embeddings.npy"
    MODEL_NAME  = "all-MiniLM-L6-v2"   # 22MB, fast CPU model

    def __init__(self):
        self.index     = None
        self.model     = None
        self.dim       = 384
        self._available = FAISS_AVAILABLE and SBERT_AVAILABLE

    def build(self, doc_texts: list[str], force_rebuild: bool = False):
        if not self._available:
            print("[INFO] FAISS/SBERT unavailable — dense retrieval disabled")
            return

        idx_path = Path(self.INDEX_PATH)
        emb_path = Path(self.EMBEDS_PATH)

        if idx_path.exists() and emb_path.exists() and not force_rebuild:
            print("[INFO] Loading existing FAISS index...")
            self.index = faiss.read_index(str(idx_path))
            self._doc_texts = doc_texts
            return

        print(f"[INFO] Building FAISS index for {len(doc_texts)} documents...")
        idx_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(self.MODEL_NAME)
        embeddings = self.model.encode(doc_texts, batch_size=64,
                                       show_progress_bar=True,
                                       convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        faiss.write_index(self.index, str(idx_path))
        np.save(str(emb_path), embeddings)
        print(f"[INFO] FAISS index built: {self.index.ntotal} vectors, dim={embeddings.shape[1]}")

    def search(self, query: str, top_k: int = 20) -> list[int]:
        if not self._available or self.index is None:
            return []
        if self.model is None:
            self.model = SentenceTransformer(self.MODEL_NAME)
        q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        _, indices = self.index.search(q_emb, top_k)
        return indices[0].tolist()


# ── Main HybridRetriever ───────────────────────────────────────────────────────
class HybridRetriever:
    """
    Three-signal hybrid retriever: BM25 + TF-IDF + FAISS (if available).
    Fused via Reciprocal Rank Fusion with stage-aware re-ranking.

    Novel vs ChatGPT:
      - Answers grounded in curated 700-example DS knowledge base
      - Three complementary signals reduce vocabulary mismatch
      - Stage-aware re-ranking aligns answers to pipeline position
    """

    def __init__(self, dataset_path: str = "data/dataset.csv",
                 build_faiss: bool = False):
        self.documents:   list[dict] = []
        self.corpus_texts: list[str] = []
        self.bm25       = None
        self.tfidf_vec  = None
        self.tfidf_mat  = None
        self.faiss_idx  = FAISSIndex()
        self._build(dataset_path, build_faiss)

    def _build(self, path: str, build_faiss: bool):
        p = Path(path)
        if not p.exists():
            print(f"[WARN] Dataset not found: {path}")
            return

        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            stage_val = row.get("pipeline_stage", row.get("stage", 1))
            try:
                row["pipeline_stage"] = int(stage_val)
            except (TypeError, ValueError):
                row["pipeline_stage"] = 1
        self.documents = rows
        self.corpus_texts = [build_doc_text(r) for r in rows]

        # BM25
        if BM25_AVAILABLE:
            tokenized = [tokenize(t) for t in self.corpus_texts]
            self.bm25 = BM25Okapi(tokenized)
            print(f"[INFO] BM25 index built: {len(rows)} documents")

        # TF-IDF
        self.tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                                          sublinear_tf=True, min_df=1)
        self.tfidf_mat = self.tfidf_vec.fit_transform(self.corpus_texts)
        print(f"[INFO] TF-IDF index built: {self.tfidf_mat.shape}")

        # FAISS dense index
        if build_faiss:
            self.faiss_idx.build(self.corpus_texts)

    # ── Individual retrievers ────────────────────────────────────────────────
    def _bm25_rank(self, query: str, k: int = 30) -> list[int]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        return np.argsort(scores)[::-1][:k].tolist()

    def _tfidf_rank(self, query: str, k: int = 30) -> list[int]:
        if self.tfidf_vec is None:
            return []
        q_vec = self.tfidf_vec.transform([query])
        sims  = cosine_similarity(q_vec, self.tfidf_mat).flatten()
        return np.argsort(sims)[::-1][:k].tolist()

    def _faiss_rank(self, query: str, k: int = 30) -> list[int]:
        return self.faiss_idx.search(query, k)

    # ── Stage re-ranking ─────────────────────────────────────────────────────
    def _stage_boost(self, doc_idx: int, target_stage: Optional[int],
                     score: float) -> float:
        if target_stage is None:
            return score
        doc_stage = self.documents[doc_idx].get("pipeline_stage", 0)
        if doc_stage == target_stage:
            return score * 1.4
        elif abs(doc_stage - target_stage) == 1:
            return score * 1.1
        return score

    # ── Main retrieve function ────────────────────────────────────────────────
    def retrieve_context(self, query: str, top_k: int = 5,
                         predicted_stage: Optional[int] = None,
                         stage_filter: Optional[int] = None) -> list[dict]:
        """
        Retrieve top-k most relevant documents using hybrid RRF.

        Args:
            query:           Natural language question.
            top_k:           Number of results to return.
            predicted_stage: Stage for score boosting (optional).
            stage_filter:    If set, only return docs from this stage.

        Returns:
            List of dicts with retrieval_score and rank added.
        """
        if not self.documents:
            return []

        # Collect rankings from all available signals
        rankings = []
        bm25_r   = self._bm25_rank(query)
        tfidf_r  = self._tfidf_rank(query)
        faiss_r  = self._faiss_rank(query)

        if bm25_r:   rankings.append(bm25_r)
        if tfidf_r:  rankings.append(tfidf_r)
        if faiss_r:  rankings.append(faiss_r)

        if not rankings:
            return []

        fused = rrf(rankings)

        # Stage boost + filter
        boosted = []
        for doc_idx, score in fused:
            if stage_filter is not None:
                if self.documents[doc_idx].get("pipeline_stage") != stage_filter:
                    continue
            boosted.append((doc_idx, self._stage_boost(doc_idx, predicted_stage, score)))

        boosted.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc_idx, score) in enumerate(boosted[:top_k], 1):
            doc = dict(self.documents[doc_idx])
            doc["retrieval_score"] = round(score, 6)
            doc["rank"]            = rank
            doc["signals_used"]    = (
                ("BM25 " if bm25_r else "") +
                ("TF-IDF " if tfidf_r else "") +
                ("FAISS" if faiss_r else "")
            ).strip()
            results.append(doc)

        return results

    # ── Convenience alias ────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 5,
                 predicted_stage: Optional[int] = None) -> list[dict]:
        """Alias for retrieve_context — maintains backward compatibility."""
        return self.retrieve_context(query, top_k, predicted_stage)

    def get_stats(self) -> dict:
        return {
            "total_documents": len(self.documents),
            "bm25_available":  BM25_AVAILABLE,
            "faiss_available": FAISS_AVAILABLE and self.faiss_idx.index is not None,
            "tfidf_vocab_size": (len(self.tfidf_vec.vocabulary_)
                                  if self.tfidf_vec else 0),
            "signals": (
                (["BM25"] if BM25_AVAILABLE else []) +
                ["TF-IDF"] +
                (["FAISS"] if FAISS_AVAILABLE and self.faiss_idx.index is not None else [])
            ),
            "stage_counts": {
                int(s): sum(1 for d in self.documents
                            if int(d.get("pipeline_stage", 0)) == int(s))
                for s in range(1, 8)
            },
        }

    def evaluate_retrieval(self, test_queries: list[dict]) -> dict:
        """Compute MRR, Recall@K on a test set."""
        rrs, r1, r3, r5 = [], [], [], []
        latencies = []
        import time

        for item in test_queries:
            t0 = time.perf_counter()
            results = self.retrieve_context(item["query"], top_k=5)
            latencies.append((time.perf_counter() - t0) * 1000)

            relevant = item["relevant_stage"]
            rr = 0.0
            for rank_idx, r in enumerate(results, 1):
                if int(r["pipeline_stage"]) == relevant:
                    rr = 1.0 / rank_idx
                    break
            rrs.append(rr)

            top_stages = [int(r["pipeline_stage"]) for r in results]
            r1.append(1.0 if relevant in top_stages[:1] else 0.0)
            r3.append(1.0 if relevant in top_stages[:3] else 0.0)
            r5.append(1.0 if relevant in top_stages[:5] else 0.0)

        return {
            "mrr":          round(float(np.mean(rrs)), 4),
            "recall_at_1":  round(float(np.mean(r1)),  4),
            "recall_at_3":  round(float(np.mean(r3)),  4),
            "recall_at_5":  round(float(np.mean(r5)),  4),
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "n_queries":    len(test_queries),
        }


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("=== HybridRetriever Demo ===\n")
    retriever = HybridRetriever("data/dataset.csv")

    print("\nSystem stats:", retriever.get_stats())

    queries = [
        ("How do I fill missing values in Age?",    4),
        ("Plot a correlation heatmap",              3),
        ("Train a Random Forest classifier",        6),
        ("What is the AUC score?",                  7),
        ("How do I one-hot encode categoricals?",   5),
    ]

    print("\n=== Retrieval Demo ===")
    for q, expected in queries:
        results = retriever.retrieve_context(q, top_k=3, predicted_stage=expected)
        print(f"\nQ: {q!r}  [expected S{expected}]")
        for r in results:
            match = "✓" if int(r["pipeline_stage"]) == expected else "✗"
            print(f"  {match} [S{r['pipeline_stage']}] {r['explanation'][:55]}…  "
                  f"score={r['retrieval_score']:.5f}  signals={r['signals_used']}")

    print("\n=== Evaluation ===")
    test_set = [{"query": q, "relevant_stage": s} for q, s in queries]
    metrics = retriever.evaluate_retrieval(test_set)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
