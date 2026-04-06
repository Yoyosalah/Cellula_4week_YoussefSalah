from src.loader import load_data
from src.splitter import split_docs
from src.embeddings import get_embedding_model
from src.vectorstore import get_or_build_vectorstore
from src.retriever import get_retriever, retrieve_with_confidence
from src.ragchain import build_explanation_ragchain, build_generation_ragchain
from src.router import build_router_chain, classify_intent
from src.learning import learn_new_function, rebuild_retriever
from src.memory import apply_memory_chain_wrapper

import gradio as gr

# ==========================================
# 1. Initialize Backend ONCE
# ==========================================

print("Initializing Backend...")

docs       = load_data()
chunks     = split_docs(docs)
embeddings = get_embedding_model()

vectorstore   = get_or_build_vectorstore(chunks, embeddings)
gen_retriever = get_retriever(vectorstore)

router_chain    = build_router_chain()
base_generate   = build_generation_ragchain(gen_retriever)
base_explain    = build_explanation_ragchain()

memory_generate = apply_memory_chain_wrapper(base_generate)
memory_explain  = apply_memory_chain_wrapper(base_explain)

SESSION_ID = "gradio_user_session"

# ==========================================
# 2. Message helpers — dict format, no type= needed
# ==========================================

def u(text: str) -> dict:
    return {"role": "user", "content": text}

def a(text: str) -> dict:
    return {"role": "assistant", "content": text}

# ==========================================
# 3. Gradio Logic Functions
# ==========================================

def process_chat(user_message: str, history: list):
    if not user_message.strip():
        return history, gr.update(), gr.update(), "", gr.update(value="")

    history = history + [u(user_message)]

    intent = classify_intent(router_chain, user_message)
    print(f"[Router] Intent classified as: {intent}")

    if intent == "Explain":
        response = memory_explain.invoke(
            {"question": user_message},
            config={"configurable": {"session_id": SESSION_ID}},
        )
        history = history + [a(response)]
        return (
            history,
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            gr.update(value=""),
        )

    # intent == "Generate"
    is_known = retrieve_with_confidence(vectorstore, user_message, threshold=0.7)

    if is_known:
        response = memory_generate.invoke(
            {"question": user_message},
            config={"configurable": {"session_id": SESSION_ID}},
        )
        history = history + [a(response)]
        return (
            history,
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            gr.update(value=""),
        )
    else:
        history = history + [a(
            "I don't have a confident answer for this. "
            "Could you please teach me using the panel below?"
        )]
        return (
            history,
            gr.update(visible=False),  # hide chat input
            gr.update(visible=True),   # show learning panel
            user_message,              # save to state
            gr.update(value=""),
        )


def handle_learning_submit(code: str, explanation: str, pending_query: str, history: list):
    global memory_generate

    if not code.strip() or not explanation.strip():
        history = history + [a("⚠️ Please provide both code and an explanation to teach me.")]
        return (
            history,
            gr.update(visible=False),
            gr.update(visible=True),
            pending_query,
            gr.update(value=code),
            gr.update(value=explanation),
        )

    learn_new_function(vectorstore, pending_query, code, explanation)

    new_retriever   = rebuild_retriever(vectorstore, k=3)
    new_base_gen    = build_generation_ragchain(new_retriever)
    memory_generate = apply_memory_chain_wrapper(new_base_gen)

    history = history + [a(
        f"✅ Successfully learned: *'{pending_query}'*. I'll use this in future answers!"
    )]
    return (
        history,
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        gr.update(value=""),
        gr.update(value=""),
    )


def handle_learning_cancel(history: list):
    history = history + [a("Learning cancelled. What else can I help you with?")]
    return (
        history,
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        gr.update(value=""),
        gr.update(value=""),
    )


# ==========================================
# 4. UI Layout
# ==========================================

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Intelligent Coding Assistant")
    gr.Markdown(
        "Ask me to **generate** Python code or **explain** programming concepts. "
        "If I don't know something, I'll ask you to teach me!"
    )

    pending_query_state = gr.State("")

    # No type= parameter — this build uses dict format natively
    chatbot = gr.Chatbot(height=500, label="Chat")

    with gr.Row(visible=True) as chat_input_row:
        msg_input = gr.Textbox(
            placeholder="e.g. Write a function to flatten a nested list...",
            scale=8,
            show_label=False,
            container=False,
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    with gr.Group(visible=False) as learning_panel:
        gr.Markdown("### 🧠 Teach Me Something New")
        gr.Markdown(
            "I didn't have a confident answer for your last request. "
            "Fill in the fields below and I'll remember it for next time."
        )
        teach_code = gr.Code(
            language="python",
            label="1. Python Code",
            interactive=True,
        )
        teach_explanation = gr.Textbox(
            label="2. Explanation",
            lines=3,
            placeholder="Briefly explain what this code does and how it works...",
        )
        with gr.Row():
            learn_btn  = gr.Button("Save to Memory", variant="primary")
            cancel_btn = gr.Button("Cancel")

    # ==========================================
    # 5. Event Wiring
    # ==========================================

    shared_inputs  = [msg_input, chatbot]
    shared_outputs = [chatbot, chat_input_row, learning_panel, pending_query_state, msg_input]

    msg_input.submit(fn=process_chat, inputs=shared_inputs, outputs=shared_outputs)
    submit_btn.click(fn=process_chat, inputs=shared_inputs, outputs=shared_outputs)

    learn_btn.click(
        fn=handle_learning_submit,
        inputs=[teach_code, teach_explanation, pending_query_state, chatbot],
        outputs=[chatbot, chat_input_row, learning_panel, pending_query_state, teach_code, teach_explanation],
    )

    cancel_btn.click(
        fn=handle_learning_cancel,
        inputs=[chatbot],
        outputs=[chatbot, chat_input_row, learning_panel, pending_query_state, teach_code, teach_explanation],
    )


if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(theme=gr.themes.Soft())