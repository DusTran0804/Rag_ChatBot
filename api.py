import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import router

app = FastAPI(
    title="RAG Chatbot API", 
    description="RESTful API cho hệ thống Multimodal RAG Chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import threading

@app.on_event("startup")
def preload_models():
    def load():
        try:
            print("Đang tải trước mô hình Embeddings...")
            from app.core.config import get_embeddings
            get_embeddings()
            print("Đang tải trước mô hình Rerank...")
            from app.service.retriever import get_rerank_model
            get_rerank_model()
            print("Tải mô hình hoàn tất.")
        except Exception as e:
            print(f"Lỗi khi tải trước mô hình: {e}")
            
    threading.Thread(target=load, daemon=True).start()

app.include_router(router, prefix="/api/v1")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", summary="Giao diện Web Chatbot")
async def serve_frontend():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    port = int(os.getenv("PORT", 8000))
    print(f"Khởi động server API tại http://localhost:{port}")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
