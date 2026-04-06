from langchain_huggingface import HuggingFaceEmbeddings
import torch

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[Embeddings] Loaded embedding model: {model_name}")
    return embeddings