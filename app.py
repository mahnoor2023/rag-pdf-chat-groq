```python
"""
Advanced RAG PDF Chat App
UI layer only.

Business logic:
- rag_engine.py -> PDF extraction, chunking, embeddings, FAISS
- groq_client.py -> Groq API and prompt management
- config.py -> application configuration
"""

import os
from datetime import datetime

import streamlit as st

from config import FALLBACK_MODELS
from rag_engine import RAGEngine
from groq_client import (
    get_client,
    fetch_available_models,
    build_prompt,
    stream_answer,
)


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG PDF Chat · Groq",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------------------------
# CUSTOM STYLE
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>

    .hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #6C5CE7 0%,
            #341F97 100%
        );
        margin-bottom: 1.4rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.3rem;
        font-size: 1.9rem;
    }

    .hero p {
        color: #e6e1ff;
        margin: 0;
        font-size: 0.95rem;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        color: white;
        font-size: 0.78rem;
        margin-right: 0.4rem;
    }

    .stChatMessage {
        border-radius: 14px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">

        <h1>📄 Chat with your PDFs</h1>

        <p>
            <span class="badge">⚡ Groq inference</span>
            <span class="badge">🔎 FAISS retrieval</span>
            <span class="badge">🧩 Open-source embeddings</span>
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


engine: RAGEngine = st.session_state.engine


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:

    st.subheader("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key from Groq Console.",
    )

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    # -------------------------------------------------------
    # MODELS
    # -------------------------------------------------------

    available_models = (
        fetch_available_models(api_key)
        if api_key
        else FALLBACK_MODELS
    )

    model = st.selectbox(
        "Model",
        available_models,
        index=0,
    )

    # -------------------------------------------------------
    # CHUNKING
    # -------------------------------------------------------

    with st.expander("🔧 Chunking & Retrieval Options"):

        chunk_size = st.slider(
            "Chunk size (words)",
            min_value=100,
            max_value=1000,
            value=400,
            step=50,
        )

        chunk_overlap = st.slider(
            "Chunk overlap (words)",
            min_value=0,
            max_value=300,
            value=50,
            step=10,
        )

        top_k = st.slider(
            "Top-k chunks to retrieve",
            min_value=1,
            max_value=6,
            value=3,
        )

    # -------------------------------------------------------
    # DOCUMENT UPLOAD
    # -------------------------------------------------------

    st.subheader("📎 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    c1, c2 = st.columns(2)

    process_clicked = c1.button(
        "📥 Process",
        use_container_width=True,
    )

    clear_clicked = c2.button(
        "🗑️ Clear",
        use_container_width=True,
    )

    # -------------------------------------------------------
    # SAVE / LOAD
    # -------------------------------------------------------

    st.divider()

    st.caption("Persistent index")

    c3, c4 = st.columns(2)

    save_clicked = c3.button(
        "💾 Save",
        use_container_width=True,
    )

    load_clicked = c4.button(
        "📂 Load",
        use_container_width=True,
    )

    # -------------------------------------------------------
    # DOCUMENT INFORMATION
    # -------------------------------------------------------

    if engine.processed_files:

        st.divider()

        m1, m2 = st.columns(2)

        m1.metric(
            "Documents",
            len(engine.processed_files),
        )

        m2.metric(
            "Chunks",
            len(engine.chunks),
        )

        for fname in engine.processed_files:
            st.caption(f"• {fname}")


# ---------------------------------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------------------------------

if process_clicked:

    if not uploaded_files:

        st.sidebar.warning(
            "Please upload at least one PDF."
        )

    else:

        with st.spinner(
            "Extracting, chunking and indexing PDFs..."
        ):

            total_new = 0

            for file in uploaded_files:

                try:

                    total_new += engine.add_pdf(
                        file,
                        chunk_size,
                        chunk_overlap,
                    )

                except Exception as e:

                    st.sidebar.error(
                        f"Error processing {file.name}: {e}"
                    )

            if total_new:

                st.sidebar.success(
                    f"Indexed {total_new} new chunks."
                )

            else:

                st.sidebar.info(
                    "No new chunks were added."
                )


# ---------------------------------------------------------------------------
# CLEAR
# ---------------------------------------------------------------------------

if clear_clicked:

    engine.clear()

    st.session_state.chat_history = []

    st.rerun()


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

if save_clicked:

    if engine.save():

        st.sidebar.success(
            "FAISS index saved successfully."
        )

    else:

        st.sidebar.warning(
            "Nothing to save yet."
        )


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

if load_clicked:

    if engine.load():

        st.sidebar.success(
            "FAISS index loaded successfully."
        )

    else:

        st.sidebar.warning(
            "No saved index found."
        )


# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------

if not api_key:

    st.info(
        "👈 Enter your Groq API key in the sidebar to get started."
    )

elif engine.index is None:

    st.info(
        "👈 Upload a PDF and click **Process** to start chatting."
    )


# ---------------------------------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------------------------------

for turn in st.session_state.chat_history:

    with st.chat_message(turn["role"]):

        st.markdown(
            turn["content"]
        )

        if turn.get("sources"):

            with st.expander("📚 Sources"):

                for source in turn["sources"]:

                    st.markdown(
                        f"**{source['source']} — "
                        f"page {source['page']}**  · "
                        f"relevance {source['score']:.2f}"
                    )

                    preview = source["text"][:300]

                    st.caption(
                        preview + "..."
                    )


# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------

query = st.chat_input(
    "Ask a question about your document(s)..."
)


if query:

    # -------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------

    if not api_key:

        st.error(
            "Please enter your Groq API key first."
        )

        st.stop()

    if engine.index is None:

        st.error(
            "Please upload and process a PDF first."
        )

        st.stop()

    # -------------------------------------------------------
    # USER MESSAGE
    # -------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # -------------------------------------------------------
    # RETRIEVE DOCUMENT CONTEXT
    # -------------------------------------------------------

    with st.spinner(
        "Searching your documents..."
    ):

        context_chunks = engine.retrieve(
            query,
            k=top_k,
        )

    if not context_chunks:

        assistant_answer = (
            "I couldn't find relevant information "
            "in the uploaded document."
        )

        with st.chat_message("assistant"):

            st.markdown(assistant_answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": assistant_answer,
                "sources": [],
            }
        )

        st.stop()

    # -------------------------------------------------------
    # BUILD PROMPT
    # -------------------------------------------------------

    system_prompt, user_prompt = build_prompt(
        query,
        context_chunks,
    )

    # -------------------------------------------------------
    # GROQ CLIENT
    # -------------------------------------------------------

    try:

        client = get_client(api_key)

    except Exception as e:

        error_message = (
            f"⚠️ Could not initialize Groq client: {e}"
        )

        with st.chat_message("assistant"):

            st.error(error_message)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": error_message,
                "sources": context_chunks,
            }
        )

        st.stop()

    # -------------------------------------------------------
    # GENERATE ANSWER
    # -------------------------------------------------------

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        try:

            # Only send recent conversation history.
            recent_history = (
                st.session_state.chat_history[:-1][-4:]
            )

            for delta in stream_answer(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                chat_history=recent_history,
            ):

                full_response += delta

                placeholder.markdown(
                    full_response + "▌"
                )

            placeholder.markdown(
                full_response
            )

        except Exception as e:

            full_response = (
                f"⚠️ Error calling Groq API: {e}"
            )

            placeholder.error(
                full_response
            )

        # ---------------------------------------------------
        # SOURCES
        # ---------------------------------------------------

        if context_chunks:

            with st.expander(
                "📚 Sources"
            ):

                for source in context_chunks:

                    st.markdown(
                        f"**{source['source']} — "
                        f"page {source['page']}**  · "
                        f"relevance "
                        f"{source['score']:.2f}"
                    )

                    st.caption(
                        source["text"][:300] + "..."
                    )

    # -------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # -------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": context_chunks,
        }
    )


# ---------------------------------------------------------------------------
# EXPORT CHAT
# ---------------------------------------------------------------------------

if st.session_state.chat_history:

    lines = []

    for turn in st.session_state.chat_history:

        role = (
            "You"
            if turn["role"] == "user"
            else "Assistant"
        )

        lines.append(
            f"{role}: {turn['content']}\n"
        )

    st.download_button(
        "⬇️ Export chat history",
        data="\n".join(lines),
        file_name=(
            f"chat_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f".txt"
        ),
        mime="text/plain",
    )
```
