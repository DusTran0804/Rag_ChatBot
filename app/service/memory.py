from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from app.core.config import get_llm
from app.core.prompt import MEMORY_CONTEXTUALIZE_PROMPT

history_store = {}

def get_session_history(session_id: str):
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

def add_message_to_history(session_id: str, role: str, content: str):
    history = get_session_history(session_id)
    if role == "user":
        history.add_user_message(content)
    else:
        history.add_ai_message(content)

def contextualize_user_query(user_query: str, session_id: str) -> str:
    history = get_session_history(session_id)
    llm = get_llm(temperature=0)

    if not history.messages:
        return user_query

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", MEMORY_CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
 
    chain = contextualize_q_prompt | llm | StrOutputParser()
    rewritten_query = chain.invoke({
        "chat_history": history.messages,
        "input": user_query
    })
    return rewritten_query