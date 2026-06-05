from langchain_core.documents import Document

from bm25_retrieval import bm25_search, index_chunks as index_bm25
from dense_retrieval import index_chunks as index_dense, vector_search
from ingestion import load_and_chunk

RRF_K = 60


def index_chunks(chunks: list[Document]) -> None:
    index_bm25(chunks)
    index_dense(chunks)


def _reciprocal_rank_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    top_k: int,
) -> list[dict]:
    rrf_scores: dict[tuple[str, int], float] = {}

    for rank, result in enumerate(bm25_results, start=1):
        key = (result["text"], result["page"])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)

    for rank, result in enumerate(dense_results, start=1):
        key = (result["text"], result["page"])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    return [
        {
            "text": text,
            "page": page,
            "score": score,
        }
        for (text, page), score in ranked
    ]


def hybrid_search(query: str, top_k: int = 20) -> list[dict]:
    bm25_results = bm25_search(query, top_k=top_k)
    dense_results = vector_search(query, top_k=top_k)
    return _reciprocal_rank_fusion(bm25_results, dense_results, top_k)


if __name__ == "__main__":
    chunks = load_and_chunk("data/sample.pdf")
    index_chunks(chunks)

    results = hybrid_search("refund policy")
    for result in results[:10]:
        print(result)
