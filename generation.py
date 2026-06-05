import os
import re

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You answer questions using only the provided document context.

Rules:
- Answer only from the provided context.
- Always cite sources inline using [1], [2], etc., matching the chunk numbers in the context.
- If the answer is not available in the context, say exactly: "I don't know based on the provided documents."
- When you cite sources, end your response with a blank line, then a "Sources:" section listing each citation you used, one per line, in the format:
[1] page 7
"""

UNKNOWN_ANSWER = "I don't know based on the provided documents."


def _parse_context_pages(context: str) -> dict[int, int]:
    return {
        int(index): int(page)
        for index, page in re.findall(r"\[(\d+)\] page (\d+)\n", context)
    }


def _append_sources(answer: str, index_to_page: dict[int, int]) -> str:
    if UNKNOWN_ANSWER in answer:
        return answer.strip()

    body = answer.split("Sources:")[0].strip()
    cited = sorted(
        {int(n) for n in re.findall(r"\[(\d+)\]", body)},
        key=int,
    )
    if not cited:
        return body

    source_lines = [
        f"[{index}] page {index_to_page[index]}"
        for index in cited
        if index in index_to_page
    ]
    if not source_lines:
        return body

    return f"{body}\n\nSources:\n" + "\n".join(source_lines)


def generate_answer(question: str, context: str) -> str:
    # Lazy imports to avoid DLL conflict with FAISS/PyTorch on Windows
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)
    answer = str(response.content) or ""
    return _append_sources(answer, _parse_context_pages(context))


if __name__ == "__main__":
    from context_assembly import build_context
    from hybrid_retrieval import hybrid_search, index_chunks
    from ingestion import load_and_chunk
    from reranking import rerank

    doc_chunks = load_and_chunk("data/sample.pdf")
    index_chunks(doc_chunks)

    query = "What is the refund period?"
    candidates = hybrid_search(query)
    reranked = rerank(query, candidates, top_k=3)
    context = build_context(reranked)

    print(generate_answer(query, context))
