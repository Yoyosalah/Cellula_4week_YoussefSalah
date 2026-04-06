from langchain_core.documents import Document
from src.splitter import split_docs
import uuid


def learn_new_function(vectorstore, query: str, code: str, explanation: str):
    page_content = f"Task Description: {query}\nExplanation: {explanation}"
    metadata = {
        "canonical_solution": code,
        "type": "self_learned",
        "task_id": f"learned_{uuid.uuid4().hex[:8]}",
    }

    doc = Document(page_content=page_content, metadata=metadata)
    chunks = split_docs([doc])
    vectorstore.add_documents(chunks)
    print("[Self-Learning] Successfully stored new function in ChromaDB.")


def rebuild_retriever(vectorstore, k: int = 3):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    print("[Self-Learning] Retriever rebuilt with updated vectorstore.")
    return retriever