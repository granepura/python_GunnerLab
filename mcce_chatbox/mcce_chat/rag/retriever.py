"""Hybrid retrieval: ChromaDB cosine similarity + BM25 keyword search."""

import os
import sys


def _load_collection(index_dir: str):
    try:
        import chromadb
    except ImportError:
        return None
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name="nomic-ai/nomic-embed-text-v1",
                                                  trust_remote_code=True)
    except ImportError:
        return None

    if not os.path.isdir(index_dir):
        return None

    client = chromadb.PersistentClient(path=index_dir)
    try:
        return client.get_collection("mcce4_source", embedding_function=ef)
    except Exception:
        return None


def _bm25_search(query: str, collection, top_k: int = 5) -> list:
    """BM25 keyword search over the collection documents."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return []

    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data["documents"]
    metas = all_data["metadatas"]
    ids = all_data["ids"]

    if not docs:
        return []

    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in ranked:
        if scores[idx] > 0:
            results.append({
                "text": docs[idx],
                "filepath": metas[idx].get("filepath", ""),
                "chunk_type": metas[idx].get("chunk_type", ""),
                "name": metas[idx].get("name", ""),
                "score_bm25": float(scores[idx]),
                "id": ids[idx],
            })
    return results


def hybrid_search(query: str, root_dir: str, index_dir: str = None, top_k: int = 6) -> list:
    """Hybrid search: cosine (top 8) + BM25 (top 5), merge and deduplicate to top_k."""
    if index_dir is None:
        index_dir = os.path.join(root_dir, "bin", ".mcce_chat_index")

    collection = _load_collection(index_dir)
    if collection is None or collection.count() == 0:
        return []

    cosine_results = collection.query(query_texts=[query], n_results=min(8, collection.count()),
                                      include=["documents", "metadatas", "distances"])

    seen_ids = set()
    merged = []

    if cosine_results and cosine_results["ids"]:
        for i, doc_id in enumerate(cosine_results["ids"][0]):
            doc = cosine_results["documents"][0][i]
            meta = cosine_results["metadatas"][0][i]
            dist = cosine_results["distances"][0][i] if cosine_results.get("distances") else 1.0
            merged.append({
                "text": doc,
                "filepath": meta.get("filepath", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "name": meta.get("name", ""),
                "score": 1.0 - dist,
                "id": doc_id,
            })
            seen_ids.add(doc_id)

    bm25_results = _bm25_search(query, collection, top_k=5)
    for r in bm25_results:
        if r["id"] not in seen_ids:
            r["score"] = r.pop("score_bm25", 0) * 0.3
            merged.append(r)
            seen_ids.add(r["id"])

    merged.sort(key=lambda x: x.get("score", 0), reverse=True)

    results = []
    for item in merged[:top_k]:
        results.append({
            "text": item["text"],
            "filepath": item["filepath"],
            "chunk_type": item["chunk_type"],
            "name": item["name"],
        })
    return results


def format_context(chunks: list) -> str:
    """Format retrieved chunks as context for the LLM prompt."""
    if not chunks:
        return ""
    parts = ["--- Retrieved MCCE4 source context ---"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"\n[{i}] {c['filepath']}  ({c['chunk_type']}: {c['name']})")
        parts.append(c["text"][:2000])
    parts.append("\n--- End of retrieved context ---")
    return "\n".join(parts)
