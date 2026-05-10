from fastapi import APIRouter, HTTPException
from app.schema.model import ChatRequest, ChatResponse
from app.service.agent import run_advanced_rag_agent

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, summary="Gửi câu hỏi cho RAG Chatbot")
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = run_advanced_rag_agent(request.query, request.session_id)
        return ChatResponse(response=response_text, session_id=request.session_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", summary="Kiểm tra trạng thái API")
async def health_check():
    return {"status": "ok", "message": "API is running"}
