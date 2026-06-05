import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ingestion import load_and_chunk

_bm25: BM25Okapi | None = None
_chunk_texts: list[str] = []
_chunk_pages: list[int] = []


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _page_number(chunk: Document) -> int:
    label = chunk.metadata.get("page_label")
    if label is not None:
        try:
            return int(label)
        except (TypeError, ValueError):
            pass
    return int(chunk.metadata.get("page", 0)) + 1


def index_chunks(chunks: list[Document]) -> None:
    global _bm25, _chunk_texts, _chunk_pages

    _chunk_texts = [chunk.page_content for chunk in chunks]
    _chunk_pages = [_page_number(chunk) for chunk in chunks]
    tokenized_corpus = [tokenize(text) for text in _chunk_texts]
    _bm25 = BM25Okapi(tokenized_corpus)


def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    if _bm25 is None:
        raise RuntimeError("BM25 index not built. Call index_chunks() first.")

    query_tokens = tokenize(query)
    scores = _bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [
        {
            "text": _chunk_texts[i],
            "page": _chunk_pages[i],
            "score": float(scores[i]),
        }
        for i in ranked_indices
    ]


if __name__ == "__main__":
    chunks = load_and_chunk("data/sample.pdf")
    index_chunks(chunks)

    results = bm25_search("refund policy")
    for result in results[:5]:
        print(result)
