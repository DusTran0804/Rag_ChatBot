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

def contextualize_user_query(user_query: str, session_id: str) -> str:
    try:
        history = get_session_history(session_id)
        messages = history.messages
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi khi đọc lịch sử hội thoại database ({e}). Sử dụng bộ nhớ RAM tạm thời.")
        if session_id not in history_store:
            history_store[session_id] = ChatMessageHistory()
        messages = history_store[session_id].messages

    if not messages:
        return user_query

    llm = get_llm(temperature=0)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", MEMORY_CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
 
    chain = contextualize_q_prompt | llm | StrOutputParser()
    from app.core.config import invoke_chain_with_retry
    rewritten_query = invoke_chain_with_retry(chain, {
        "chat_history": messages,
        "input": user_query
    })
    return rewritten_query