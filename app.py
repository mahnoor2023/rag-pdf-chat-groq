"""
Advanced RAG (Retrieval-Augmented Generation) PDF Chat App — UI layer only.
Business logic lives in rag_engine.py (FAISS/embeddings) and groq_client.py (LLM calls),
so this file stays focused on the Streamlit interface.
"""

import os
from datetime import datetime

import streamlit as st

from config import FALLBACK_MODELS
from rag_engine import RAGEngine
from groq_client import get_client, fetch_available_models, build_prompt, stream_answer

# ---------------------------------------------------------------------------
# PAGE CONFIG & CUSTOM STYLE
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG PDF Chat · Groq", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #6C5CE7 0%, #341F97 100%);
        margin-bottom: 1.4rem;
    }
    .hero h1 { color: white; margin-bottom: 0.3rem; font-size: 1.9rem; }
    .hero p { color: #e6e1ff; margin: 0; font-size: 0.95rem; }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        color: white;
        font-size: 0.78rem;
        margin-right: 0.4rem;
    }
    .stChatMessage { border-radius: 14px; }
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

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
        help="Get a free key at console.groq.com/keys",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    available_models = fetch_available_models(api_key) if api_key else FALLBACK_MODELS
    model = st.selectbox("Model", available_models, index=0)

    with st.expander("🔧 Chunking options"):
        chunk_size = st.slider("Chunk size (words)", 100, 1000, 400, step=50)
        chunk_overlap = st.slider("Chunk overlap (words)", 0, 300, 50, step=10)
        top_k = st.slider("Top-k chunks to retrieve", 1, 10, 4)

    st.subheader("📎 Documents")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    c1, c2 = st.columns(2)
    process_clicked = c1.button("📥 Process", use_container_width=True)
    clear_clicked = c2.button("🗑️ Clear", use_container_width=True)

    st.divider()
    st.caption("Persistent index")
    c3, c4 = st.columns(2)
    save_clicked = c3.button("💾 Save", use_container_width=True)
    load_clicked = c4.button("📂 Load", use_container_width=True)

    if engine.processed_files:
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Documents", len(engine.processed_files))
        m2.metric("Chunks", len(engine.chunks))
        for fname in engine.processed_files:
            st.caption(f"• {fname}")

        with st.expander("🔍 Debug: search indexed text"):
            debug_query = st.text_input("Find a word/phrase in indexed chunks", key="debug_search")
            if debug_query:
                matches = [c for c in engine.chunks if debug_query.lower() in c["text"].lower()]
                st.caption(f"{len(matches)} chunk(s) contain '{debug_query}'")
                for m in matches[:10]:
                    st.markdown(f"**{m['source']} — page {m['page']}**")
                    st.code(m["text"], language=None)

# --- handle sidebar actions -------------------------------------------------
if process_clicked and uploaded_files:
    with st.spinner("Extracting and indexing PDFs..."):
        total_new = 0
        for file in uploaded_files:
            total_new += engine.add_pdf(file, chunk_size, chunk_overlap)
        if total_new:
            st.sidebar.success(f"Indexed {total_new} new chunks.")
        else:
            st.sidebar.info("Nothing new to index.")

if clear_clicked:
    engine.clear()
    st.session_state.chat_history = []
    st.rerun()

if save_clicked:
    st.sidebar.success("Saved to disk.") if engine.save() else st.sidebar.warning("Nothing to save yet.")

if load_clicked:
    st.sidebar.success("Loaded from disk.") if engine.load() else st.sidebar.warning("No saved index found.")

# ---------------------------------------------------------------------------
# MAIN CHAT AREA
# ---------------------------------------------------------------------------
if not api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started.")
elif engine.index is None:
    st.info("👈 Upload a PDF and click **Process** to start chatting.")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("📚 Sources"):
                for s in turn["sources"]:
                    st.markdown(f"**{s['source']} — page {s['page']}**  ·  relevance {s['score']:.2f}")
                    st.caption(s["text"][:300] + "...")

query = st.chat_input("Ask a question about your document(s)...")

if query:
    if not api_key:
        st.error("Please enter your Groq API key first.")
    elif engine.index is None:
        st.error("Please upload and process a PDF first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        context_chunks = engine.retrieve(query, k=top_k)
        system_prompt, user_prompt = build_prompt(query, context_chunks)
        client = get_client(api_key)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                for delta in stream_answer(
                    client, model, system_prompt, user_prompt, st.session_state.chat_history[:-1]
                ):
                    full_response += delta
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"⚠️ Error calling Groq API: {e}"
                placeholder.markdown(full_response)

            if context_chunks:
                with st.expander("📚 Sources"):
                    for s in context_chunks:
                        st.markdown(f"**{s['source']} — page {s['page']}**  ·  relevance {s['score']:.2f}")
                        st.caption(s["text"][:300] + "...")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response, "sources": context_chunks}
        )

# --- export chat -------------------------------------------------------
if st.session_state.chat_history:
    lines = [
        f"{'You' if t['role'] == 'user' else 'Assistant'}: {t['content']}\n"
        for t in st.session_state.chat_history
    ]
    st.download_button(
        "⬇️ Export chat history",
        data="\n".join(lines),
        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )
