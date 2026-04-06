import os
from langchain_chroma import Chroma

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = str(BASE_DIR /"chroma_db")

def build_vectorstore(chunks, embeddings):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    print(f"[Vectorstore] Built and persisted vectorstore at '{VECTORSTORE_DIR}'")
    return vectorstore


def load_vectorstore(embeddings):
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(f"Vectorstore not found at '{VECTORSTORE_DIR}'. Run build first.")
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )
    print(f"[Vectorstore] Loaded existing vectorstore from '{VECTORSTORE_DIR}'")
    return vectorstore


def get_or_build_vectorstore(chunks, embeddings):
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        return load_vectorstore(embeddings)
    return build_vectorstore(chunks, embeddings)