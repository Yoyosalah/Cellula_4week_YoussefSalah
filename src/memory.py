from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages

_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

def get_memory_trimmer(max_messages: int = 10):
    return trim_messages(
        max_tokens=max_messages,
        strategy="last",
        token_counter=len,
        include_system=True,
    )

def apply_memory_chain_wrapper(chain, input_key: str = "question"):
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key=input_key,
        history_messages_key="history",
    )
    print("[Memory] Chain successfully wrapped with session memory.")
    return chain_with_memory