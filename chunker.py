# chunker.py
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Simple character-based chunker with overlap.
    Returns list of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        # advance
        start = end - overlap
    return chunks
