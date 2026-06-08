"""Task 10: answer generation with citations."""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

TOP_K = 5
TOP_P = 0.9  # Focused nucleus sampling for factual RAG if an API model is used.
TEMPERATURE = 0.3  # Low temperature keeps citation answers conservative.

SYSTEM_PROMPT = """Answer in Vietnamese using only the provided context.
Every factual claim must include a bracketed citation.
If evidence is insufficient, say: I cannot verify this information."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return chunks
    front = [chunks[i] for i in range(0, len(chunks), 2)]
    back = [chunks[i] for i in range(len(chunks) - 1, 0, -2)]
    return front + back


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _offline_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I cannot verify this information"
    source = chunks[0].get("metadata", {}).get("source", "Source")
    first_sentence = chunks[0]["content"].split(".")[0].strip()
    if not first_sentence:
        return "I cannot verify this information"
    return (
        f"Du lieu lien quan nhat cho cau hoi '{query}' la: "
        f"{first_sentence}. [{source}]"
    )


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = ""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or ""
        except Exception:
            answer = ""

    if not answer:
        answer = _offline_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
    }


if __name__ == "__main__":
    print(generate_with_citation("Hinh phat ma tuy?")["answer"])
