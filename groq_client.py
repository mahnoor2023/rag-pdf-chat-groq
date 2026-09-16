```python
"""Everything related to talking to the Groq API."""

from groq import Groq

from config import FALLBACK_MODELS


# ---------------------------------------------------------------------------
# LIMITS
# ---------------------------------------------------------------------------

# Maximum amount of retrieved PDF text sent to the LLM.
# This prevents context_length_exceeded errors.
MAX_CONTEXT_CHARS = 9000

# Maximum number of previous chat messages sent to the LLM.
MAX_HISTORY_TURNS = 4

# Maximum characters allowed for one previous chat message.
MAX_HISTORY_CHARS = 2000


# ---------------------------------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------------------------------

def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# AVAILABLE MODELS
# ---------------------------------------------------------------------------

def fetch_available_models(api_key: str):
    """Try to pull the live model list from Groq; fall back to a static list."""
    try:
        client = get_client(api_key)

        ids = [m.id for m in client.models.list().data]

        # Drop non-chat models such as speech-to-text, TTS and moderation.
        chat_ids = [
            m
            for m in ids
            if not any(
                x in m.lower()
                for x in ["whisper", "tts", "guard"]
            )
        ]

        return sorted(chat_ids) if chat_ids else FALLBACK_MODELS

    except Exception:
        return FALLBACK_MODELS


# ---------------------------------------------------------------------------
# CONTEXT BUILDER
# ---------------------------------------------------------------------------

def build_prompt(query, context_chunks):
    """
    Build a controlled RAG prompt.

    Only a limited amount of retrieved PDF text is sent to Groq.
    This helps prevent context_length_exceeded errors.
    """

    context_parts = []
    total_chars = 0

    for c in context_chunks:
        source = c.get("source", "Unknown")
        page = c.get("page", "Unknown")
        text = c.get("text", "")

        source_header = f"[Source: {source} - Page {page}]\n"

        remaining = MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        # Keep the source header even when trimming the chunk.
        available_text = remaining - len(source_header)

        if available_text <= 0:
            break

        text = text[:available_text]

        context_parts.append(
            f"{source_header}{text}"
        )

        total_chars += len(source_header) + len(text)

    context = "\n\n".join(context_parts)

    system = (
        "You are a helpful RAG assistant. "
        "Answer the user's question using only the provided PDF context. "
        "Do not invent information. "
        "If the answer is not present in the context, clearly say that "
        "the information was not found in the uploaded documents. "
        "When answering, mention the relevant source file and page number "
        "when that information is available."
    )

    user = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question concisely and accurately."
    )

    return system, user


# ---------------------------------------------------------------------------
# STREAMING ANSWER
# ---------------------------------------------------------------------------

def stream_answer(
    client,
    model,
    system_prompt,
    user_prompt,
    chat_history
):
    """
    Stream the answer from Groq while keeping chat history small.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Keep only the most recent few messages.
    recent_history = chat_history[-MAX_HISTORY_TURNS:]

    for turn in recent_history:
        role = turn.get("role")
        content = turn.get("content", "")

        if role not in ["user", "assistant"]:
            continue

        # Prevent old long responses from consuming the context window.
        content = content[:MAX_HISTORY_CHARS]

        messages.append(
            {
                "role": role,
                "content": content
            }
        )

    # Current RAG question + retrieved context.
    messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    stream = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        temperature=0.3,
        max_tokens=800,
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta.content

        if delta:
            yield delta
```
