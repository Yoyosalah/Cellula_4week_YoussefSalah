from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from pathlib import Path

from src.system_prompt import GENERATE_SYSTEM_PROMPT, EXPLAIN_SYSTEM_PROMPT
from src.memory import get_memory_trimmer

load_dotenv(Path(__file__).resolve().parent.parent / "app.env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Check your app.env path.")


def format_docs(docs):
    return "\n\n".join(doc.metadata["canonical_solution"] for doc in docs)


def build_generation_ragchain(
    retriever, model_name: str = "arcee-ai/trinity-large-preview:free"
):
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    trimmer = get_memory_trimmer(max_messages=10)

    # RunnablePassthrough.assign ensures all keys exist in the dict
    # before the prompt tries to render them.
    # history is trimmed here; context is retrieved here;
    # question passes through unchanged.
    rag_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs,
            history=itemgetter("history") | trimmer,
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"[RAG Chain] Built GENERATE_RAG chain with model '{model_name}'")
    return rag_chain


def build_explanation_ragchain(
    model_name: str = "arcee-ai/trinity-large-preview:free"
):
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXPLAIN_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    trimmer = get_memory_trimmer(max_messages=10)

    # For explain, there is no retriever — history is trimmed,
    # question passes through. The memory wrapper injects history
    # automatically via the MessagesPlaceholder key.
    explain_chain = (
        RunnablePassthrough.assign(
            history=itemgetter("history") | trimmer,
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"[RAG Chain] Built EXPLAIN_RAG chain with model '{model_name}'")
    return explain_chain