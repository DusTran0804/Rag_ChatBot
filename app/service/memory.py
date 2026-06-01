from langchain_community.chat_message_histories import ChatMessageHistory, SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from app.core.config import get_llm, SUPABASE_DATABASE_URL
from app.core.prompt import MEMORY_CONTEXTUALIZE_PROMPT

# Bộ lưu trữ dự phòng trong RAM khi không kết nối được Database
history_store = {}

def get_session_history(session_id: str):
    if SUPABASE_DATABASE_URL:
        try:
            return SQLChatMessageHistory(
                session_id=session_id,
                connection=SUPABASE_DATABASE_URL,
                table_name="chat_history"
            )
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khởi tạo Supabase Database ({e}). Sử dụng bộ nhớ RAM tạm thời.")
            
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

def add_message_to_history(session_id: str, role: str, content: str):
    try:
        history = get_session_history(session_id)
        if role == "user":
            history.add_user_message(content)
        else:
            history.add_ai_message(content)
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi khi ghi tin nhắn vào lịch sử database ({e}). Quay lại dùng RAM.")
        if session_id not in history_store:
            history_store[session_id] = ChatMessageHistory()
        history = history_store[session_id]
        if role == "user":
            history.add_user_message(content)
        else:
            history.add_ai_message(content)

def contextualize_user_query(raw_query: str, session_id: str) -> str:
    history = get_session_history(session_id)
    if not history.messages:
        return raw_query

    llm = get_llm(temperature=0.0)
    
    # Extract only the last 4 messages to avoid token limit and distraction
    recent_messages = history.messages[-4:]
    history_str = "\n".join([f"{'User' if m.type == 'human' else 'AI'}: {m.content}" for m in recent_messages])
    
    prompt_text = f"""BẠN LÀ MỘT CÔNG CỤ XỬ LÝ NGÔN NGỮ (TEXT PROCESSOR). BẠN KHÔNG PHẢI LÀ CHATBOT!
TUYỆT ĐỐI KHÔNG ĐƯỢC TRẢ LỜI CÂU HỎI. NHIỆM VỤ DUY NHẤT LÀ TRẢ VỀ MỘT CHUỖI VĂN BẢN (TEXT).

Lịch sử trò chuyện gần đây:
{history_str}

Nhiệm vụ: Phân tích đầu vào hiện tại của người dùng là "{raw_query}"
1. Nếu đầu vào này chứa các từ phụ thuộc ngữ cảnh (như "nó", "vậy còn cái này"), hãy viết lại thành một câu hỏi hoàn chỉnh dựa vào Lịch sử.
2. Nếu đầu vào này ĐÃ RÕ NGHĨA hoặc CHỈ LÀ MỘT CỤM TỪ KHÓA , BẮT BUỘC TRẢ VỀ Y HỆT NGUYÊN VĂN CỤM TỪ ĐÓ, không được bịa thành câu hỏi.
3. CHỈ IN RA KẾT QUẢ CUỐI CÙNG. KHÔNG TRẢ LỜI CÂU HỎI, KHÔNG GIẢI THÍCH, KHÔNG CHÀO HỎI."""

    try:
        from langchain_core.messages import HumanMessage
        result = llm.invoke([HumanMessage(content=prompt_text)])
        contextualized_q = result.content.strip()
        # Fallback in case LLM is too chatty (though this prompt should strictly prevent it)
        if len(contextualized_q) > len(raw_query) * 3 and len(contextualized_q) > 50:
            # If the output is absurdly long, it probably hallucinated an answer. Fallback to raw.
            if "?" not in contextualized_q:
                return raw_query
        return contextualized_q
    except Exception as e:
        import traceback
        traceback.print_exc()
        return raw_query