from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi của người dùng")
    session_id: str = Field(default="default_session", description="ID của phiên chat")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Câu trả lời từ chatbot")
    session_id: str = Field(..., description="ID của phiên chat")
