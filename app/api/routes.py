import os
import json
import glob
import logging
import threading
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schema.model import ChatRequest, ChatResponse
from app.service.agent import run_advanced_rag_agent

logger = logging.getLogger(__name__)

router = APIRouter()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

eval_lock = threading.Lock()
eval_status = {"status": "idle", "message": "Sẵn sàng"}

def run_evaluation_task(limit: int = None):
    global eval_status
    with eval_lock:
        eval_status["status"] = "running"
        eval_status["message"] = "Đang chạy đánh giá RAGAS (sử dụng Gemini)..."
        
    try:
        from evaluation.evaluate import main as run_evaluation_main
        run_evaluation_main(limit=limit)
        with eval_lock:
            eval_status["status"] = "idle"
            eval_status["message"] = "Hoàn thành đánh giá gần nhất thành công"
    except Exception as e:
        logger.error(f"Lỗi khi chạy đánh giá: {e}")
        with eval_lock:
            eval_status["status"] = "idle"
            eval_status["message"] = f"Thất bại: {str(e)}"

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

@router.post("/evaluation/run", summary="Chạy quy trình đánh giá RAGAS")
async def run_evaluation(background_tasks: BackgroundTasks, limit: int = None):
    global eval_status
    with eval_lock:
        if eval_status["status"] == "running":
            return {"status": "running", "message": "Quy trình đánh giá khác hiện đang chạy."}
    
    background_tasks.add_task(run_evaluation_task, limit)
    return {"status": "running", "message": "Đã bắt đầu chạy đánh giá RAGAS trong background."}

@router.get("/evaluation/status", summary="Kiểm tra trạng thái quy trình đánh giá")
async def get_evaluation_status():
    global eval_status
    return eval_status

@router.get("/evaluation/latest", summary="Lấy kết quả đánh giá mới nhất")
async def get_latest_evaluation(timestamp: str = None):
    try:
        results_dir = os.path.join(ROOT_DIR, "evaluation", "results")
        if timestamp:
            summary_path = os.path.join(results_dir, f"ragas_summary_{timestamp}.json")
            results_path = os.path.join(results_dir, f"ragas_results_{timestamp}.json")
        else:
            summary_path = os.path.join(results_dir, "ragas_summary_latest.json")
            results_path = os.path.join(results_dir, "ragas_results_latest.json")
        
        if not os.path.exists(summary_path) or not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail="Chưa có kết quả đánh giá nào cho lần chạy này.")
            
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            
        return {
            "summary": summary,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluation/history", summary="Lấy danh sách các lần đánh giá lịch sử")
async def get_evaluation_history():
    try:
        results_dir = os.path.join(ROOT_DIR, "evaluation", "results")
        files = glob.glob(os.path.join(results_dir, "ragas_summary_*.json"))
        history = []
        for file in files:
            if "latest" in file:
                continue
            try:
                with open(file, "r", encoding="utf-8") as f:
                    history.append(json.load(f))
            except Exception as e:
                logger.warning(f"Lỗi khi đọc file lịch sử {file}: {e}")

        history = sorted(history, key=lambda x: x.get("evaluation_timestamp", ""), reverse=True)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
