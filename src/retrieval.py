import os, json, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client import QdrantClient
from .config import QDRANT_URL, QDRANT_API_KEY, COLLECTION, EMBEDDING_MODEL
from .ingest import CHUNKS_CACHE
from .reranker import rerank_candidates

encoder = SentenceTransformer(EMBEDDING_MODEL)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def load_chunks_cache():
    if not os.path.exists(CHUNKS_CACHE):
        return []
    with open(CHUNKS_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)

def build_bm25():
    cache = load_chunks_cache()
    corpus = [c["text"] for c in cache]
    tokenized = [c.split() for c in corpus]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return bm25, cache

def dense_search_qdrant(query: str, top_k:int=10):
    qvec = encoder.encode([query])[0].tolist()
    try:
        hits = qdrant.search(collection_name=COLLECTION, query_vector=qvec, limit=top_k)
    except Exception as e:
        print("Qdrant search error:", e)
        return []
    out = []
    for h in hits:
        payload = h.payload
        out.append({"id": h.id, "score": h.score, "text": payload.get("text"), "meta": {k:payload.get(k) for k in payload if k!="text"}})
    return out

def bm25_search(query: str, top_k:int=10):
    bm25, cache = build_bm25()
    if bm25 is None:
        return []
    tokenized = query.split()
    top_n = bm25.get_top_n(tokenized, [c["text"] for c in cache], n=top_k)
    results = []
    for t in top_n:
        idx = next((i for i,c in enumerate(cache) if c["text"]==t), None)
        if idx is not None:
            results.append({"id": f"bm25_{idx}", "score": None, "text": cache[idx]["text"], "meta": cache[idx]["meta"]})
    return results

def hybrid_retrieve(query: str, top_k_dense=5, top_k_bm25=5, rerank=False):
    dense = dense_search_qdrant(query, top_k=top_k_dense)
    sparse = bm25_search(query, top_k=top_k_bm25)
    seen = {}
    for d in dense + sparse:
        key = d.get("id") or d.get("text")[:60]
        if key not in seen:
            seen[key] = d
        else:
            if d.get("score") is not None:
                seen[key]["score"] = max(seen[key].get("score", 0) or 0, d.get("score"))
    candidates = list(seen.values())
    for c in candidates:
        if c.get("score") is None:
            c["score"] = 0.0
    candidates.sort(key=lambda x: x["score"], reverse=True)
    if rerank and len(candidates)>1:
        candidates = rerank_candidates(query, candidates)
    return candidates
