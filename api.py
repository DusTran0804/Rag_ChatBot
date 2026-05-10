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

app.include_router(router, prefix="/api/v1")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", summary="Giao diện Web Chatbot")
async def serve_frontend():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    print("Khởi động server API tại http://localhost:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
