from app.service.memory import contextualize_user_query, add_message_to_history
from app.service.router import logical_router
from app.service.retriever import hybrid_retrieval, apply_post_retrieval_rerank
from app.service.generator import generate_answer
from app.core.config import get_llm

def run_advanced_rag_agent(raw_user_query: str, session_id: str):
    llm = get_llm()
    full_query = contextualize_user_query(raw_user_query, session_id)
    routing_res = logical_router(full_query)
    raw_docs = hybrid_retrieval(routing_res)
    final_context = apply_post_retrieval_rerank(full_query, raw_docs)
    answer = generate_answer(full_query, final_context, llm)
    add_message_to_history(session_id, "user", raw_user_query)
    add_message_to_history(session_id, "ai", answer) 
    return answer