# 📄 RAG PDF Chat (Groq + FAISS)

Chat with your PDFs using an open-source RAG pipeline and Groq's fast open-weight LLMs.

## Project structure
```
.
├── app.py              # Streamlit UI (chat interface, sidebar, styling)
├── rag_engine.py        # PDF extraction, chunking, embeddings, FAISS index
├── groq_client.py       # Groq API calls (model list, prompt, streaming)
├── config.py             # Constants / settings
├── requirements.txt
├── .streamlit/
│   └── config.toml      # Custom dark theme
└── README.md
```

## Features
- Multiple PDF upload & indexing
- Adjustable chunk size / overlap
- Semantic search via FAISS + sentence-transformers embeddings
- Streaming answers from Groq (with model picker)
- Chat memory (follow-up questions)
- Source citations (file + page number)
- Persistent index (save/load FAISS index to disk)
- Export chat history as .txt

## Run locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"   # or paste it in the sidebar
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → select the repo, branch, and `app.py` as the main file.
4. Click **Deploy**.
5. Once live, paste your Groq API key in the app's sidebar (get one free at
   https://console.groq.com/keys).

## Note on persistence
Streamlit Cloud's filesystem is ephemeral — the saved FAISS index will be lost
if the app restarts or redeploys. The Save/Load buttons are useful within a
running session, not as permanent cloud storage.
