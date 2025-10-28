# init_db.py
import os
import psycopg

DSN="postgresql://postgres:YourPassword@localhost:5432/YourDatabaseName"

def init():
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id bigserial PRIMARY KEY,
                source TEXT,
                metadata JSONB,
                text_chunk TEXT,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT now()
            );
            """)
            # create HNSW index (concurrent not used in small local db)
            cur.execute("CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops);")
            conn.commit()

if __name__ == "__main__":
    init()
    print("DB initialized.")
