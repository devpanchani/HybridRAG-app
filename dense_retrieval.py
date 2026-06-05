import chromadb
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_chroma_client = None
_collection = None


def _page_number(chunk: Document) -> int:
    label = chunk.metadata.get("page_label")
    if label is not None:
        try:
            return int(label)
        except (TypeError, ValueError):
            pass
    return int(chunk.metadata.get("page", 0)) + 1


def _init_chroma():
    global _chroma_client, _collection, _model
    if _chroma_client is None:
        # Create persistent storage folder inside data/
        persist_dir = os.path.join(os.getcwd(), "data", "chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
        _collection = _chroma_client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)


def clear_index() -> None:
    """Wipe all chunks from ChromaDB so a new PDF can be indexed from scratch."""
    _init_chroma()
    _collection.delete(where={"page": {"$gte": 0}})
    print("ChromaDB collection cleared. Ready to index a new document.")


def index_chunks(chunks: list[Document]) -> None:
    """
    Persist document chunks to ChromaDB. 
    If the database already contains chunks, we skip re-embedding to save time!
    """
    _init_chroma()
    
    # Simple check: If collection has items, assume we've already indexed the PDF.
    if _collection.count() > 0:
        print(f"ChromaDB already has {_collection.count()} chunks. Skipping heavy embedding process!")
        return

    _chunk_texts = [chunk.page_content for chunk in chunks]
    _chunk_pages = [_page_number(chunk) for chunk in chunks]
    _ids = [str(i) for i in range(len(chunks))]

    print("Generating dense embeddings (this only happens once now)...")
    embeddings = _model.encode(_chunk_texts, convert_to_numpy=True).astype(np.float32).tolist()
    
    metadatas = [{"page": page} for page in _chunk_pages]
    
    print("Saving embeddings to persistent ChromaDB...")
    _collection.add(
        ids=_ids,
        embeddings=embeddings,
        documents=_chunk_texts,
        metadatas=metadatas
    )


def vector_search(query: str, top_k: int = 20) -> list[dict]:
    _init_chroma()
    
    query_embedding = _model.encode([query], convert_to_numpy=True).astype(np.float32).tolist()
    
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    out = []
    if results['documents'] and len(results['documents']) > 0:
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for i in range(len(docs)):
            out.append({
                "text": docs[i],
                "page": metas[i]["page"],
                # Convert cosine distance to a similarity score
                "score": 1.0 - distances[i],
            })
            
    return out


if __name__ == "__main__":
    from ingestion import load_and_chunk
    
    chunks = load_and_chunk("data/sample.pdf")
    index_chunks(chunks)

    print("\nSearch Results:")
    results = vector_search("refund policy", top_k=3)
    for result in results:
        print(result)
