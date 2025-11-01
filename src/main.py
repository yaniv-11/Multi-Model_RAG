import os, shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from .ingest import index_file
from .retrieval import hybrid_retrieve
from .graph import build_graph, query_graph_for_entities
from dotenv import load_dotenv
load_dotenv()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Multimodal RAG - Assignment Deliverable", version="0.1")

@app.get("/")
def root():
    return {"status":"ok", "message":"Multimodal RAG API"}

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())
    background_tasks.add_task(index_file, str(dest), file.filename)
    return JSONResponse({"status":"accepted", "filename": file.filename})

@app.post("/query")
async def query(q: str, top_k:int=6, rerank:bool=False):
    try:
        cand = hybrid_retrieve(q, top_k_dense=top_k, top_k_bm25=top_k, rerank=rerank)
        out = []
        for c in cand[:top_k]:
            out.append({
                "text": c.get("text"),
                "meta": c.get("meta"),
                "score": c.get("score")
            })
        return {"query": q, "results": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/build_graph")
def build_triplet_graph(triplets_path: str = "triplets.json"):
    try:
        G = build_graph(triplets_path)
        return {"status":"ok", "nodes": len(G.nodes()), "edges": len(G.edges())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph_query")
def graph_query(entities: list):
    facts = query_graph_for_entities(entities)
    return {"facts": facts}
