from datasets import load_dataset
from langchain_core.documents import Document


def load_data():
    print("[Loader] Loading dataset...")
    ds = load_dataset("openai/openai_humaneval",split="test")

    docs = []
    for d in ds:
        doc = Document(
            page_content=d['prompt'],
            metadata = {
            'task_id': d['task_id'],
            'canonical_solution': d['canonical_solution']
            }
        )
        docs.append(doc)
    
    return docs


if __name__ == "__main__":
    docs = load_data()
    print("Page Content:",docs[0].page_content)
    print('Metadata (task_id):',docs[0].metadata['task_id'])
    print('Metadata (canonical_solution):',docs[0].metadata['canonical_solution'])