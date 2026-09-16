"""Everything related to talking to the Groq API."""

from groq import Groq

from config import FALLBACK_MODELS


def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def fetch_available_models(api_key: str):
    """Try to pull the live model list from Groq; fall back to a static list."""
    try:
        client = get_client(api_key)
        ids = [m.id for m in client.models.list().data]
        # Drop non-chat models (speech-to-text, moderation, etc.) heuristically
        chat_ids = [m for m in ids if not any(x in m.lower() for x in ["whisper", "tts", "guard"])]
        return sorted(chat_ids) if chat_ids else FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


def build_prompt(query, context_chunks):
    context = "\n\n".join(
        f"[Source: {c['source']} - Page {c['page']}]\n{c['text']}"
        for c in context_chunks
    )
    system = (
        "You are a helpful assistant that answers questions strictly using the "
        "provided context extracted from the user's PDF document(s). "
        "If the answer is not contained in the context, say clearly that you "
        "don't have enough information from the documents, and STOP there — "
        "do not substitute or mention unrelated information just to fill the answer. "
        "Never guess or fill gaps with information about a different entity, company, "
        "or topic than what was asked. "
        "When you do answer, mention which source file and page the information came from."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return system, user


def stream_answer(client, model, system_prompt, user_prompt, chat_history):
    """Yields text deltas from Groq's streaming chat completion, with chat memory."""
    messages = [{"role": "system", "content": system_prompt}]
    for turn in chat_history[-6:]:  # last few turns = chat memory
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_prompt})

    stream = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        temperature=0.3,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
