from typing import List, Dict

def rerank_candidates(query: str, candidates: List[Dict], top_k=10):
    # Simple heuristic: boost candidates containing query tokens
    qtokens = set(query.lower().split())
    scored = []
    for c in candidates:
        text = c.get("text","").lower()
        exact = sum(1 for w in qtokens if w in text)
        c_score = (c.get("score",0) or 0) + exact*0.15
        c["score_rerank"]=c_score
        scored.append(c)
    scored.sort(key=lambda x: x["score_rerank"], reverse=True)
    return scored[:top_k]
