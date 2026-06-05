def build_context(chunks: list[dict]) -> str:
    blocks = [
        f"[{index}] page {chunk['page']}\n{chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


if __name__ == "__main__":
    from hybrid_retrieval import hybrid_search, index_chunks
    from ingestion import load_and_chunk
    from reranking import rerank

    chunks = load_and_chunk("data/sample.pdf")
    index_chunks(chunks)

    candidates = hybrid_search("refund policy")
    reranked = rerank("refund policy", candidates, top_k=3)
    context = build_context(reranked)

    print(context)
