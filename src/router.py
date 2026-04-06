from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from pathlib import Path
from src.system_prompt import ROUTER_SYSTEM_PROMPT

load_dotenv(Path(__file__).resolve().parent.parent / "app.env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Check your app.env path.")


def build_router_chain(model_name: str = "arcee-ai/trinity-large-preview:free"):
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0, # deterministic classification
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    router_chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(f"[Router] Built router chain with model '{model_name}'")
    return router_chain


def classify_intent(router_chain, query: str) -> str:
    result = router_chain.invoke(query)
    
    intent = result.strip()
    
    if intent not in ("Explain", "Generate"):
        print(f"[Router Warning] Unexpected intent: '{intent}', defaulting to 'Generate'")
        return "Generate"
    
    return intent