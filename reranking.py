from sentence_transformers import CrossEncoder

from hybrid_retrieval import hybrid_search, index_chunks
from ingestion import load_and_chunk

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(MODEL_NAME)
    return _cross_encoder


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    return [
        {
            "text": candidate["text"],
            "page": candidate["page"],
            "rerank_score": float(score),
        }
        for candidate, score in ranked
    ]


if __name__ == "__main__":
    chunks = load_and_chunk("data/sample.pdf")
    index_chunks(chunks)

    candidates = hybrid_search("refund policy")
    results = rerank("refund policy", candidates, top_k=3)

    for result in results:
        print(result)
