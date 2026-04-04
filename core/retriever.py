import math
import collections
from typing import List, Dict, Any

class HybridRetriever:
    """BM25-first hybrid retriever scaffold with stage-aware boosting."""
    
    def __init__(self, use_dense: bool = False, use_cross_encoder: bool = False, stage_boost: float = 1.3):
        self.use_dense = use_dense
        self.use_cross_encoder = use_cross_encoder
        self.stage_boost = stage_boost
        
        self.documents = []
        self.tokenized_corpus = []
        self.doc_freqs = []
        self.avgdl = 0.0
        self.corpus_size = 0
        self._bm25 = None
        self._dense_model = None
        self._doc_embeddings = None
        self._cross_encoder = None

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Builds naive BM25 index on the fly."""
        self.documents = []
        for doc in documents:
            prepared = dict(doc)
            prepared["query"] = str(prepared.get("query", ""))
            prepared["answer"] = str(prepared.get("answer", ""))
            prepared["code"] = str(prepared.get("code", ""))
            try:
                prepared["stage"] = int(prepared.get("stage", 1))
            except (TypeError, ValueError):
                prepared["stage"] = 1
            self.documents.append(prepared)

        self.tokenized_corpus = [doc["query"].lower().split() for doc in self.documents]
        self.corpus_size = len(documents)
        
        # Calculate BM25 params
        self.doc_freqs = []
        lengths = 0
        for tokens in self.tokenized_corpus:
            freq_dict = collections.Counter(tokens)
            self.doc_freqs.append(freq_dict)
            lengths += len(tokens)
            
        self.avgdl = lengths / float(self.corpus_size) if self.corpus_size > 0 else 0
        self._build_optional_indexes()

    def _build_optional_indexes(self) -> None:
        self._bm25 = None
        if self.tokenized_corpus:
            try:
                from rank_bm25 import BM25Okapi
                self._bm25 = BM25Okapi(self.tokenized_corpus)
            except Exception:
                self._bm25 = None

        if self.use_dense:
            try:
                from sentence_transformers import SentenceTransformer
                self._dense_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._doc_embeddings = self._dense_model.encode(
                    [doc["query"] for doc in self.documents],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            except Exception:
                self._dense_model = None
                self._doc_embeddings = None

        if self.use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception:
                self._cross_encoder = None
        
    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        k1 = 1.5
        b = 0.75
        
        freqs = self.doc_freqs[doc_idx]
        doc_len = len(self.tokenized_corpus[doc_idx])
        
        for token in set(query_tokens):
            # document frequency of the token
            df = sum(1 for d in self.doc_freqs if token in d)
            if df == 0:
                continue
                
            idf = math.log(1 + (self.corpus_size - df + 0.5) / (df + 0.5))
            f = freqs.get(token, 0)
            
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * (doc_len / self.avgdl))
            score += idf * (numerator / denominator)
            
        return score

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank + 1)

    def retrieve(
        self,
        query: str,
        active_stage: int = -1,
        top_k: int = 3,
        skill_level: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Returns top matching docs with retrieval scores."""
        if not self.documents:
            return []
            
        query_tokens = query.lower().split()
        sparse_scores = {}
        dense_scores = {}

        if self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(query_tokens)
            sparse_scores = {i: float(score) for i, score in enumerate(bm25_scores)}
        else:
            for i in range(len(self.documents)):
                sparse_scores[i] = float(self._bm25_score(query_tokens, i))

        if self.use_dense and self._dense_model is not None and self._doc_embeddings is not None:
            query_emb = self._dense_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            for i in range(len(self.documents)):
                dense_scores[i] = float(query_emb @ self._doc_embeddings[i])

        if dense_scores:
            sparse_ranked = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)
            dense_ranked = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
            fused = collections.defaultdict(float)
            for rank, (idx, _) in enumerate(sparse_ranked):
                fused[idx] += self._rrf(rank)
            for rank, (idx, _) in enumerate(dense_ranked):
                fused[idx] += self._rrf(rank)
            combined_scores = dict(fused)
        else:
            combined_scores = dict(sparse_scores)

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        picked = []
        for recency_rank, (idx, score) in enumerate(ranked):
            doc = dict(self.documents[idx])
            if active_stage > 0 and doc.get("stage") == active_stage:
                score *= self.stage_boost

            # Lightweight personalized reranking:
            # beginner -> boost explanatory answers
            # advanced -> boost code-heavy answers
            answer_len = len(str(doc.get("answer", "")).split())
            code_len = len(str(doc.get("code", "")).split())
            expl_score = min(1.0, answer_len / 60.0)
            code_score = min(1.0, code_len / 40.0)
            skill_match = (1.0 - skill_level) * expl_score + skill_level * code_score
            recency_bonus = 1.0 / float(recency_rank + 1)
            final_score = (0.7 * float(score)) + (0.2 * skill_match) + (0.1 * recency_bonus)

            doc["retrieval_score"] = float(final_score)
            doc["skill_match_score"] = float(skill_match)
            picked.append(doc)
            if len(picked) >= max(top_k * 3, top_k):
                break

        if self.use_cross_encoder and self._cross_encoder is not None and picked:
            pairs = [[query, d.get("answer", "")] for d in picked]
            rerank_scores = self._cross_encoder.predict(pairs)
            for i, score in enumerate(rerank_scores):
                picked[i]["rerank_score"] = float(score)
                picked[i]["retrieval_score"] = float(score)
            picked.sort(key=lambda x: x["retrieval_score"], reverse=True)
        else:
            picked.sort(key=lambda x: x["retrieval_score"], reverse=True)

        return picked[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        stage_counts = collections.Counter(doc.get("stage", 1) for doc in self.documents)
        return {
            "total_documents": len(self.documents),
            "bm25_available": self.corpus_size > 0,
            "stages_present": sorted(stage_counts.keys()),
            "stage_distribution": dict(stage_counts),
            "dense_enabled": self.use_dense,
            "cross_encoder_enabled": self.use_cross_encoder,
        }
