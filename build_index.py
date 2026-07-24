#!/usr/bin/env python3
"""Build/refresh the FAISS knowledge-base index with Cohere embeddings.

Splits knowledge_base.md into chunks (keeping line ranges as metadata for source
attribution, same idea as app28.py), embeds them with Cohere embed-v4.0, and
saves a FAISS index into the DEFAULT_INDEX folder (default: ./nepse_kb).

Usage:
    python build_index.py                 # uses knowledge_base.md -> nepse_kb/
    python build_index.py my_notes.md     # custom source file
"""
import os
import sys

from config import get_embeddings, DEFAULT_INDEX, HAS_COHERE

HERE = os.path.dirname(os.path.abspath(__file__))


def load_chunks(path: str):
    """Split markdown into section chunks, tracking source line ranges."""
    from langchain_core.documents import Document
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    src = os.path.basename(path)
    chunks, buf, start = [], [], 1
    for i, line in enumerate(lines, 1):
        # New "## " header starts a new chunk (keep the intro before the first).
        if line.startswith("## ") and buf:
            text = "".join(buf).strip()
            if text:
                chunks.append(Document(page_content=text,
                                       metadata={"source": src,
                                                 "line_start": start,
                                                 "line_end": i - 1}))
            buf, start = [], i
        buf.append(line)
    text = "".join(buf).strip()
    if text:
        chunks.append(Document(page_content=text,
                               metadata={"source": src, "line_start": start,
                                         "line_end": len(lines)}))
    return chunks


def main():
    if not HAS_COHERE:
        sys.exit("COHERE_API_KEY not set — add it to .env first.")

    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "knowledge_base.md")
    if not os.path.exists(src):
        sys.exit(f"Source file not found: {src}")

    docs = load_chunks(src)
    print(f"Loaded {len(docs)} chunks from {os.path.basename(src)}")

    from langchain_community.vectorstores import FAISS
    store = FAISS.from_documents(docs, embedding=get_embeddings())

    out = DEFAULT_INDEX if os.path.isabs(DEFAULT_INDEX) else os.path.join(
        HERE, DEFAULT_INDEX)
    store.save_local(out)
    print(f"Saved FAISS index to {out}/  (index.faiss + index.pkl)")


if __name__ == "__main__":
    main()
