```python
"""
Everything related to talking to the Groq API.

This module:
- Creates the Groq client
- Fetches available models
- Builds controlled RAG prompts
- Limits context size
- Limits chat history
- Streams the final answer
"""

from groq import Groq

from config import FALLBACK_MODELS


# ---------------------------------------------------------------------------
# SAFETY LIMITS
# ---------------------------------------------------------------------------

# Maximum PDF context sent to the LLM.
MAX_CONTEXT_CHARS = 9000

# Maximum number of previous chat messages.
MAX_HISTORY_MESSAGES = 4

# Maximum characters from one previous message.
MAX_HISTORY_MESSAGE_CHARS = 2000

# Maximum generated answer length.
MAX_OUTPUT_TOKENS = 800


# ---------------------------------------------------------------------------
# CLIENT
# ---------------------------------------------------------------------------

def get_client(api_key: str) -> Groq:
    """
    Create and return a Groq client.
    """

    if not api_key:

        raise ValueError(
            "Groq API key is missing."
        )

    return Groq(
        api_key=api_key
    )


# ---------------------------------------------------------------------------
# AVAILABLE MODELS
# ---------------------------------------------------------------------------

def fetch_available_models(api_key: str):
    """
    Try to fetch currently available Groq models.

    If the API request fails, use fallback models.
    """

    try:

        client = get_client(api_key)

        response = client.models.list()

        ids = [
            model.id
            for model in response.data
        ]

        # Remove non-chat models.
        chat_ids = [
            model_id
            for model_id in ids
            if not any(
                word in model_id.lower()
                for word in [
                    "whisper",
                    "tts",
                    "guard",
                ]
            )
        ]

        return (
            sorted(chat_ids)
            if chat_ids
            else FALLBACK_MODELS
        )

    except Exception:

        return FALLBACK_MODELS


# ---------------------------------------------------------------------------
# CONTEXT BUILDER
# ---------------------------------------------------------------------------

def build_prompt(query, context_chunks):
    """
    Build a controlled RAG prompt.

    Retrieved chunks are limited by character count
    before being sent to Groq.
    """

    context_parts = []

    total_chars = 0

    for chunk in context_chunks:

        source = chunk.get(
            "source",
            "Unknown source",
        )

        page = chunk.get(
            "page",
            "Unknown page",
        )

        text = chunk.get(
            "text",
            "",
        )

        if not text:
            continue

        header = (
            f"[Source: {source} - Page {page}]\n"
        )

        remaining = (
            MAX_CONTEXT_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        available_text = (
            remaining
            - len(header)
        )

        if available_text <= 0:
            break

        # Limit this chunk.
        text = text[:available_text]

        context_parts.append(
            header + text
        )

        total_chars += (
            len(header)
            + len(text)
        )

    # Join retrieved chunks.
    context = "\n\n".join(
        context_parts
    )

    # -------------------------------------------------------
    # SYSTEM PROMPT
    # -------------------------------------------------------

    system = """
You are a helpful Retrieval-Augmented Generation assistant.

Your job is to answer questions using ONLY the information
provided in the PDF context.

Rules:

1. Do not invent facts.
2. Do not use information that is not present in the context.
3. If the answer is not available in the context, say:
   "I couldn't find that information in the uploaded document."
4. Give concise and direct answers.
5. When possible, mention the source file and page number.
6. For questions about a candidate, person, organization,
   education, experience, skills, etc., extract the relevant
   information directly from the provided context.
""".strip()

    # -------------------------------------------------------
    # USER PROMPT
    # -------------------------------------------------------

    user = f"""
PDF CONTEXT:

{context}

USER QUESTION:

{query}

INSTRUCTION:

Answer the user's question using only the PDF context above.
Keep the answer concise and factual.
""".strip()

    return system, user


# ---------------------------------------------------------------------------
# STREAM ANSWER
# ---------------------------------------------------------------------------

def stream_answer(
    client,
    model,
    system_prompt,
    user_prompt,
    chat_history,
):
    """
    Stream the Groq response.

    Chat history is intentionally limited so that
    the total request remains within the model context window.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    # -------------------------------------------------------
    # RECENT CHAT HISTORY
    # -------------------------------------------------------

    recent_history = (
        chat_history[-MAX_HISTORY_MESSAGES:]
    )

    for turn in recent_history:

        role = turn.get(
            "role"
        )

        content = turn.get(
            "content",
            "",
        )

        if role not in [
            "user",
            "assistant",
        ]:
            continue

        if not content:
            continue

        # Limit old messages.
        content = content[
            :MAX_HISTORY_MESSAGE_CHARS
        ]

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    # -------------------------------------------------------
    # CURRENT QUESTION + RAG CONTEXT
    # -------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    # -------------------------------------------------------
    # GROQ REQUEST
    # -------------------------------------------------------

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.3,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    # -------------------------------------------------------
    # STREAM DELTAS
    # -------------------------------------------------------

    for chunk in stream:

        if not chunk.choices:
            continue

        delta = (
            chunk.choices[0]
            .delta
            .content
        )

        if delta:

            yield delta
```
