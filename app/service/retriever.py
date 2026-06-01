from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import QDRANT_PATH, RERANK_MODEL_NAME, get_embeddings, COLLECTION_NAME

_rerank_model = None
_qdrant_client = None

def get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        _rerank_model = CrossEncoder(RERANK_MODEL_NAME, device='cpu')
    return _rerank_model


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client

def hybrid_retrieval(routing_result: dict):
    queries = routing_result.get("queries", [])
    target_col = COLLECTION_NAME
    
    client = get_qdrant_client()
    embeddings = get_embeddings()
    all_docs_content = []

    for q in queries:
        query_vector = embeddings.embed_query(q)
        try:
            search_result = client.query_points(
                collection_name=target_col,
                query=query_vector,
                using="dense",
                limit=3
            ).points

            for point in search_result:
                content = point.payload.get('page_content', '')
                if content:
                    all_docs_content.append(content)
        except Exception as e:
            import traceback
            traceback.print_exc()

    unique_contexts = []
    seen = set()
    for text in all_docs_content:
        if text not in seen:
            unique_contexts.append(text)
            seen.add(text)
    return "\n\n---\n\n".join(unique_contexts)

def apply_post_retrieval_rerank(user_query: str, retrieved_texts: str, top_n: int = 3):
    doc_list = retrieved_texts.split("\n\n---\n\n")
    doc_list = [d for d in doc_list if d.strip()]
    
    if not doc_list:
        return "Không tìm thấy ngữ cảnh phù hợp."

    rerank_model = get_rerank_model()
    pairs = [[user_query, doc] for doc in doc_list]
    scores = rerank_model.predict(pairs)
    
    # Xử lý trường hợp model trả về mảng 2 chiều (logits cho [irrelevant, relevant])
    if len(scores.shape) > 1 and scores.shape[1] > 1:
        scores = scores[:, 1]
    elif len(scores.shape) > 1 and scores.shape[1] == 1:
        scores = scores.flatten()

    scored_docs = sorted(zip(scores.tolist(), doc_list), key=lambda x: x[0], reverse=True)
    final_docs = [doc for score, doc in scored_docs[:top_n]]

    return "\n\n---\n\n".join(final_docs)