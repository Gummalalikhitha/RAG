# embed_store.py
import os
import json
import uuid
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from chunker import chunk_text
from pypdf import PdfReader

DSN="postgresql://postgres:YourPassword@localhost:5432/YourDatabaseName"

# register vector type with psycopg
register_vector()

# load embedding model (downloads first time)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

def text_from_pdf(path):
    reader = PdfReader(path)
    s = []
    for p in reader.pages:
        try:
            s.append(p.extract_text())
        except Exception:
            pass
    return "\n".join(filter(None, s))

def store_document(source_name: str, text: str, metadata: dict = None):
    metadata = metadata or {}
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    # embed in batches to be faster
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings):
                cur.execute(
                    "INSERT INTO documents (source, metadata, text_chunk, embedding) VALUES (%s, %s, %s, %s)",
                    (source_name, json.dumps(metadata), chunk, list(map(float, emb)))
                )
        conn.commit()
    return len(chunks)

if __name__ == "__main__":
    # quick CLI sample
    import sys
    path = sys.argv[1]
    if path.lower().endswith(".pdf"):
        text = text_from_pdf(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    cnt = store_document(source_name=path, text=text)
    print(f"Inserted {cnt} chunks.")
