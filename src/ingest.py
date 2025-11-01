import os, io, json
import fitz
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from typing import List, Dict
from .config import QDRANT_URL, QDRANT_API_KEY, COLLECTION, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, USE_TESSERACT
from .utils import now_iso
import textwrap

# local cache path
CHUNKS_CACHE = "data/chunks_cache.json"
os.makedirs("data", exist_ok=True)

encoder = SentenceTransformer(EMBEDDING_MODEL)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def ensure_collection():
    try:
        col = qdrant.get_collection(COLLECTION, ignore_missing=True)
        if col is None:
            qdrant.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=qm.VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=qm.Distance.COSINE)
            )
    except Exception as e:
        print("Qdrant error:", e)
        raise

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    if not text or not text.strip():
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
        i += size - overlap
    return chunks

def ocr_image_bytes(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(img)

def extract_pages_from_pdf(path: str) -> List[Dict]:
    doc = fitz.open(path)
    pages = []
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        text = page.get_text("text")
        images = []
        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            images.append(base["image"])
        pages.append({"page": pno+1, "text": text, "images": images})
    return pages

def _append_chunks_cache(records):
    cache = []
    if os.path.exists(CHUNKS_CACHE):
        try:
            with open(CHUNKS_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except:
            cache = []
    for r in records:
        cache.append({"text": r["text"], "meta": r["meta"]})
    with open(CHUNKS_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def index_file(path: str, source: str=None) -> int:
    ensure_collection()
    if source is None:
        source = os.path.basename(path)
    _, ext = os.path.splitext(path.lower())

    records = []
    if ext == ".pdf":
        pages = extract_pages_from_pdf(path)
        for p in pages:
            text = p.get("text","")
            for cid, chunk in enumerate(chunk_text(text)):
                meta = {"source": source, "type": "text", "page": p["page"], "chunk": cid, "ts": now_iso()}
                records.append({"text": chunk, "meta": meta})
            for img_idx, img_bytes in enumerate(p.get("images", [])):
                try:
                    ocr_text = ocr_image_bytes(img_bytes) if USE_TESSERACT else ""
                    if ocr_text.strip():
                        for cid, chunk in enumerate(chunk_text(ocr_text)):
                            meta = {"source": source, "type": "ocr", "page": p["page"], "img_idx": img_idx, "chunk": cid, "ts": now_iso()}
                            records.append({"text": chunk, "meta": meta})
                except Exception as e:
                    print("OCR failed:", e)
    elif ext in [".png", ".jpg", ".jpeg"]:
        with open(path, "rb") as f:
            bytes_img = f.read()
        ocr_text = ocr_image_bytes(bytes_img) if USE_TESSERACT else ""
        for cid, chunk in enumerate(chunk_text(ocr_text)):
            meta = {"source": source, "type": "ocr", "chunk": cid, "ts": now_iso()}
            records.append({"text": chunk, "meta": meta})
    elif ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        for cid, chunk in enumerate(chunk_text(txt)):
            meta = {"source": source, "type": "text", "chunk": cid, "ts": now_iso()}
            records.append({"text": chunk, "meta": meta})
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not records:
        return 0

    texts = [r["text"] for r in records]
    embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    points = []
    for i, r in enumerate(records):
        pid = f"{source}__{i}"
        payload = {"text": r["text"], **r["meta"]}
        points.append(qm.PointStruct(id=pid, vector=embeddings[i].tolist(), payload=payload))

    B = 64
    for i in range(0, len(points), B):
        qdrant.upsert(collection_name=COLLECTION, points=points[i:i+B])

    _append_chunks_cache(records)
    return len(records)
