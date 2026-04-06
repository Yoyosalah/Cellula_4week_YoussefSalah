# 🤖 Intelligent Coding Assistant

An advanced, locally-hosted AI coding assistant built with LangChain, ChromaDB, and Gradio. This application uses Retrieval-Augmented Generation (RAG) to write, explain, and learn Python code, featuring a dynamic conversational memory and a robust modular architecture.

**Author:** Youssef Mohamed Salah 
**Program:** Mechatronics & Robotics Engineering

---

## ✨ Features

* **Smart Intent Routing:** Uses an LLM-based router to automatically classify user prompts as either "Generate" or "Explain," ensuring the correct RAG pipeline is triggered without manual toggling.
* **Retrieval-Augmented Generation (RAG):** Leverages a local ChromaDB vector store and HuggingFace embeddings to retrieve verified code snippets as context for the LLM.
* **Conversational Memory:** Maintains session-aware chat history using LangChain's `RunnableWithMessageHistory`. It employs intelligent message trimming to prevent token overflow while strictly protecting system prompts.
* **Interactive Self-Learning:** If the assistant encounters a request it isn't confident about, the Gradio UI seamlessly transforms to reveal a "Teaching Panel." Users can input custom code and explanations, which are dynamically embedded and saved into ChromaDB for future retrieval.
* **Modular Architecture:** Built with professional software engineering principles. The core logic is decoupled into a Python package (`src/`), separating UI concerns from backend LLM pipelines.

---

## 🛠️ Tech Stack

* **UI Framework:** Gradio
* **LLM Orchestration:** LangChain (LCEL)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **LLM Provider:** OpenRouter API (`arcee-ai/trinity-large-preview:free`)
* **Environment:** Designed to run seamlessly alongside complex environments (e.g., ROS 2 workspaces).

---

## 📂 Project Structure

```text
├── app.py                  # Main Gradio application and UI layout
├── app.env                 # Environment variables (API Keys)
├── requirements.txt        # Project dependencies
├── chroma_db/              # Local persistent vector database (Auto-generated)
└── src/                    # Backend Core Logic Package
    ├── __init__.py         # Package facade for clean app.py imports
    ├── embeddings.py       # HuggingFace embedding model initialization
    ├── memory.py           # Session history and message trimming logic
    ├── rag_chains.py       # Generate/Explain LCEL pipelines & retrievers
    ├── router.py           # Intent classification chain
    ├── learning.py    # Logic for dynamically updating the vector store
    ├── splitter.py         # Text splitting for new document ingestion
    ├── system_prompt.py    # Centralized LLM system instructions
    └── vectorstore.py      # ChromaDB initialization and loading
```
## 🚀 Installation & Setup
### 1. Clone or Navigate to the Project Directory
Ensure you are in the root folder containing app.py.

### 2. Install Dependencies
It is recommended to use a virtual environment. Install the core requirements:

```Bash
pip install -r requirements.txt
```
### 3. Configure Environment Variables
Create an app.env file in the root directory and add your OpenRouter API key:
```Code snippet
OPENROUTER_API_KEY=your_api_key_here
```
## 💻 Usage
Run the application from the terminal:

```Bash
python app.py
```
The Gradio interface will launch. Open the provided local URL (typically `http://127.0.0.1:7860`) in your web browser.

To Generate Code: Ask the assistant to write a function or script (e.g., "Write a ROS 2 publisher in Python").

To Explain Code: Ask the assistant to explain a concept or follow up on generated code (e.g., "Explain how the timer callback works in that script").

To Teach: If the assistant lacks the context to generate confident code, use the pop-up code editor to teach it. It will remember the logic for all future sessions.