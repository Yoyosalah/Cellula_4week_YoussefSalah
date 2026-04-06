from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_docs(docs):
    # note: splitter won't be much used since it will contain the whole prompt individually with no problem.
    # However, it's main purpose is for later on new data entering the vector database
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators=['\n\n','\n',' ']
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    from loader import load_data
    
    docs = load_data()
    splitted_docs = split_docs(docs)
    print(f"{len(docs)=}")
    print(f"{len(splitted_docs)=}")
    print("="*50)
    print(docs[0])
    print("-"*50)
    print(splitted_docs[0])