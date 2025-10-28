#app.py
import os
import json
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from chunker import chunk_text
from pypdf import PdfReader
from io import BytesIO

# ---------------- CONFIG ----------------

DSN="postgresql://YourUsername:YourPassword@localhost:5432/YourDatabaseName"


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

app = FastAPI(title="RAG with pgvector UI")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = SentenceTransformer(MODEL_NAME)

def text_from_pdf_bytes(bts):
    pdf_stream = BytesIO(bts)      # ✅ Convert bytes to a file-like object
    reader = PdfReader(pdf_stream) # ✅ Works now
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Register pgvector for embeddings
def get_connection():
    conn = psycopg.connect(DSN)
    register_vector(conn)
    return conn

# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_document(source: str = Form(...), metadata: str = Form("{}"), file: UploadFile = File(...)):
    content = await file.read()
    if file.filename.lower().endswith(".pdf"):
        text = text_from_pdf_bytes(content)
    else:
        text = content.decode("utf-8")

    metadata_obj = json.loads(metadata)
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

    conn = get_connection()
    with conn.cursor() as cur:
        for chunk, emb in zip(chunks, embeddings):
            cur.execute(
                "INSERT INTO documents (source, metadata, text_chunk, embedding) VALUES (%s, %s, %s, %s)",
                (source, json.dumps(metadata_obj), chunk, list(map(float, emb)))
            )
    conn.commit()
    conn.close()
    return {"inserted_chunks": len(chunks)}

@app.post("/query/")
async def query(q: str = Form(...), top_k: int = Form(3)):
    q_emb = model.encode([q], show_progress_bar=False, convert_to_numpy=True)[0]
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT text_chunk, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (list(map(float, q_emb)), list(map(float, q_emb)), top_k))
        rows = cur.fetchall()
    conn.close()
    results = [{"text_chunk": r[0], "similarity": float(r[1])} for r in rows]
    return {"query": q, "results": results}

