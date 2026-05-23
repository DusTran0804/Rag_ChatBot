
from app.service.memory import contextualize_user_query, add_message_to_history
from app.service.router import logical_router
from app.service.retriever import hybrid_retrieval, apply_post_retrieval_rerank
from app.service.generator import generate_answer
from app.core.config import get_llm


def run_rag_pipeline(question: str, session_id: str = "eval_session") -> dict:
    """
    Chạy toàn bộ pipeline RAG và trả về answer + contexts.

    Args:
        question: Câu hỏi của người dùng.
        session_id: ID phiên để quản lý memory.

    Returns:
        dict với các key:
            - question (str): Câu hỏi gốc
            - answer (str): Câu trả lời từ RAG pipeline
            - contexts (list[str]): Danh sách các đoạn context đã retrieve được
    """
    llm = get_llm()

    # Bước 1: Contextualize query với lịch sử hội thoại
    full_query = contextualize_user_query(question, session_id)

    # Bước 2: Router phân loại độ phức tạp và nguồn dữ liệu
    routing_res = logical_router(full_query)

    # Bước 3: Retrieve tài liệu liên quan
    raw_docs = hybrid_retrieval(routing_res)

    # Bước 4: Rerank để lấy các context tốt nhất
    final_context_str = apply_post_retrieval_rerank(full_query, raw_docs)

    # Tách chuỗi context thành danh sách
    contexts = [
        c.strip()
        for c in final_context_str.split("\n\n---\n\n")
        if c.strip()
    ]

    # Bước 5: Sinh câu trả lời
    answer = generate_answer(full_query, final_context_str, llm)

    # Lưu vào memory
    add_message_to_history(session_id, "user", question)
    add_message_to_history(session_id, "ai", answer)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts if contexts else ["Không tìm thấy ngữ cảnh phù hợp."],
    }
