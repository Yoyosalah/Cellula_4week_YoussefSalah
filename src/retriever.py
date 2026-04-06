def get_retriever(vectorstore, k: int = 3):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    print(f"[Retriever] Retriever configured with top-{k} results")
    return retriever


def retrieve_with_confidence(vectorstore, query: str, threshold: float = 0.7):
    results = vectorstore.similarity_search_with_score(query, k=1)

    if not results:
        return False

    best_doc, best_score = results[0]
    print(f"[Retriever] Best match score (L2): {best_score:.4f} | threshold: {threshold}")
    return best_score < threshold

if __name__ == "__main__":
    from loader import load_data
    from splitter import split_docs
    from embeddings import get_embedding_model
    from vectorstore import get_or_build_vectorstore

    docs = load_data()
    chunks = split_docs(docs)
    embedding_model = get_embedding_model()
    vector_store = get_or_build_vectorstore(chunks,embedding_model)
    retriever = get_retriever(vector_store)

    test_query = "Write a python function to check if a number is prime."

    retrieved_docs = retriever.invoke(test_query)
    
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"--- Result {i} ---")
        print(f"Metadata: {doc.metadata}") 
        print(f"Page_content:\n{doc.page_content}\n")

    print(retrieve_with_confidence(vector_store,test_query))
