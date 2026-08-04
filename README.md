---
title: Rag ChatBot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Advanced Enterprise RAG Chatbot 

Hệ thống Chatbot RAG (Retrieval-Augmented Generation) thông minh, xây dựng trên kiến trúc **multi-step pipeline** với khả năng:
- **Query Rewrite** — tự động viết lại câu hỏi dựa trên lịch sử hội thoại
- **Logical Router** — phân loại câu hỏi và điều hướng đến nguồn dữ liệu phù hợp
- **Sub-query Decomposition** — phân rã câu hỏi phức tạp thành nhiều sub-query
- **Hybrid Retrieval + Re-ranking** — truy xuất đa luồng kết hợp chấm điểm lại Cross-Encoder
- **Persistent Memory** — lưu lịch sử hội thoại lên Supabase (PostgreSQL) hoặc RAM

---

## Sơ đồ Kiến trúc Hệ thống

```mermaid
flowchart TD
    %% Ingestion Pipeline
    subgraph Ingestion ["Offline Data Ingestion"]
        direction LR
        A1((Actor)) --> B1[MD5 Checkpoint]
        B1 --> B2[Load Document\n.pdf .docx .txt .csv .md]
        B2 --> C1[Semantic Chunker\nbkai-vi-encoder]
        C1 --> D1[Embedding\n768-dim vectors]
    end

    D1 -- "Lưu vào Qdrant\n(dense vectors)" --> VDB[(Qdrant\nVector DB)]

    %% Query Pipeline
    U1((Client)) --> Q1([Raw Query])

    Q1 --> MEM["Memory Contextualize\nQuery Rewrite nếu có lịch sử"]
    MEM --> QR["Logical Router\nphân loại: SIMPLE / COMPLEX\nchỉ định index: ky_thuat / doanh_nghiep / chung"]

    QR -- "SIMPLE" --> RET
    QR -- "COMPLEX" --> QT["Sub-query Decomposition\ntách thành ≤3 sub-queries"]
    QT --> RET

    RET["Hybrid Retrieval"] --> RRK["Re-ranking\nCross-Encoder scoring"]

    RRK --> GEN["Generator"]
    GEN --> ANS([Answer])

    ANS --> DB[("Chat History\nSupabase / RAM fallback")]
```

---

## Cấu trúc Dự án

```
Rag_Chatbot/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints (chat, evaluation, health)
│   ├── core/
│   │   ├── config.py           # Cấu hình chung: model names, paths, LLM/Embed factory
│   │   └── prompt.py           # Tất cả system prompts (router, transform, generator, memory)
│   ├── schema/
│   │   └── model.py            # Pydantic schemas: ChatRequest, ChatResponse
│   └── service/
│       ├── agent.py            # Orchestrator — điều phối toàn bộ pipeline RAG
│       ├── memory.py           # Query rewrite + lưu/đọc lịch sử (Supabase / RAM)
│       ├── router.py           # Logical Router: phân loại độ phức tạp & chỉ định index
│       ├── query_transform.py  # Sub-query Decomposition (Multi-step)
│       ├── retriever.py        # Hybrid Retrieval + Cross-Encoder Re-ranking
│       ├── generator.py        # Sinh câu trả lời với LLaMA 3.3 70B
│       └── ingest.py           # Nạp tài liệu vào Qdrant (Semantic Chunking)
├── evaluation/
│   ├── evaluate.py             # RAGAS evaluation: Faithfulness, Context Recall
│   ├── run_eval.py             # Script chạy evaluation đầu-cuối
│   ├── rag_pipeline.py         # Wrapper pipeline để đánh giá (trả về contexts)
│   ├── optimization_dataset.json  # Bộ câu hỏi + ground truth tiếng Việt
│   └── results/               # Kết quả CSV, JSON, biểu đồ PNG
├── scripts/
│   ├── test_rag.py             # Script test nhanh RAG pipeline
│   └── clear_db.py            # Xóa toàn bộ Vector DB
├── static/                    # Web UI (HTML/CSS/JS)
├── data/                      # Thư mục chứa tài liệu đầu vào
├── VectorDB/                  # Qdrant lưu trữ local
├── api.py                     # Entry point FastAPI app
├── main.py                    # Entry point CLI chatbot
├── run.sh                     # Script menu tương tác
├── Dockerfile                 # Docker image (Python 3.12-slim, port 7860)
├── render.yaml                # Cấu hình triển khai Render.com
└── requirements.txt           # Thư viện Python
```

---

## Chi tiết các Module

### `app/service/memory.py` — Bộ nhớ & Query Rewrite

Quản lý lịch sử hội thoại theo `session_id` với **2 chế độ lưu trữ**:

| Chế độ | Điều kiện | Mô tả |
|---|---|---|
| **Supabase (PostgreSQL)** | `SUPABASE_DATABASE_URL` được cấu hình | Lưu bền vững qua `SQLChatMessageHistory`, bảng `chat_history` |
| **RAM Fallback** | Không có DB URL hoặc kết nối thất bại | Lưu tạm thời trong dict `history_store` |

**Query Rewrite**: Trước khi xử lý, LLM phân tích câu hỏi và lịch sử gần nhất (4 tin cuối). Nếu câu hỏi phụ thuộc ngữ cảnh (chứa đại từ như "nó", "cái đó"), LLM tự động viết lại thành câu độc lập hoàn chỉnh.

---

### `app/service/router.py` — Logical Router

Sử dụng **Structured Output** của LangChain để phân loại câu hỏi thành 2 chiều:

```python
class RouterLogic(BaseModel):
    logic_type: Literal["COMPLEX", "SIMPLE"]   
    target_index: Literal["ky_thuat", "doanh_nghiep", "chung"]  
```

- **COMPLEX** → kích hoạt Sub-query Decomposition
- **SIMPLE** → truy xuất trực tiếp
- Routing lỗi → fallback về `chung` + query gốc

---

### `app/service/query_transform.py` — Sub-query Decomposition

Với câu hỏi phức tạp, LLM phân rã thành tối đa **3 sub-query** độc lập, mỗi sub-query được dùng để truy xuất riêng biệt → tăng độ bao phủ ngữ cảnh.

---

### `app/service/retriever.py` — Hybrid Retrieval + Re-ranking

**Retrieval**:
- Embed từng sub-query bằng `bkai-foundation-models/vietnamese-bi-encoder`
- Tìm kiếm Qdrant với `dense` vector (cosine similarity), `top_k=3` mỗi query
- Deduplicate kết quả từ tất cả sub-query

**Re-ranking**:
- Mô hình: `amberoad/bert-multilingual-passage-reranking-msmarco` (Cross-Encoder)
- Chấm điểm từng cặp `(query, document)` → chọn **top-3** context chất lượng cao nhất
- Xử lý cả 1D và 2D logit output từ Cross-Encoder

---

### `app/service/ingest.py` — Nạp tài liệu

Hỗ trợ đọc các định dạng:

| Định dạng | Loader |
|---|---|
| `.pdf` | `PyPDFLoader` |
| `.docx` / `.doc` | `Docx2txtLoader` |
| `.txt` | `TextLoader` |
| `.csv` | `CSVLoader` |
| `.md` | `UnstructuredMarkdownLoader` |

**Tính năng MD5 Checkpoint**: Hệ thống tính toán và lưu mã băm (MD5) của các file đã nạp vào `VectorDB/ingest_checkpoint.json`. Ở các lần chạy sau, hệ thống sẽ tự động đối chiếu và bỏ qua các file chưa bị thay đổi nội dung, giúp tiết kiệm tối đa thời gian embedding và ngăn chặn việc nạp trùng lặp vào Vector DB.

Sử dụng **Semantic Chunker** (`langchain-experimental`) với ngưỡng `breakpoint_threshold_amount=0.8` để chia tài liệu dựa trên nghĩa, không phải độ dài cố định.

---

### `app/service/generator.py` — Sinh câu trả lời

Nhận `(query, top-3 context)` → gọi LLM (`llama-3.3-70b-versatile` via Groq) theo system prompt được định nghĩa trong `app/core/prompt.py`. Tích hợp retry tự động (tối đa 10 lần, exponential backoff tối đa 60s).

---

### `app/api/routes.py` — REST API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/chat` | Gửi câu hỏi, nhận câu trả lời |
| `GET` | `/health` | Kiểm tra trạng thái API |
| `POST` | `/evaluation/run` | Kích hoạt RAGAS evaluation (chạy background) |
| `GET` | `/evaluation/status` | Kiểm tra trạng thái đang chạy evaluation |
| `GET` | `/evaluation/latest` | Lấy kết quả evaluation mới nhất |
| `GET` | `/evaluation/history` | Lấy danh sách tất cả lần evaluation |


## Stack

| Thành phần | Công nghệ | Chi tiết |
|---|---|---|
| **LLM chính** | Groq `llama-3.3-70b-versatile` | Sinh câu trả lời, Router, Query Transform, Rewrite |
| **LLM đánh giá** | Groq `llama-3.1-8b-instant` | Chấm điểm RAGAS |
| **Embedding** | `bkai-foundation-models/vietnamese-bi-encoder` | Vector 768 chiều, tối ưu tiếng Việt |
| **Re-ranker** | `amberoad/bert-multilingual-passage-reranking-msmarco` | Cross-Encoder chấm điểm relevance |
| **Vector DB** | Qdrant (local) | Lưu tại `VectorDB/`, dense cosine search |
| **Chat History** | Supabase PostgreSQL / RAM | Persistent với SQLAlchemy + psycopg2 |
| **Backend** | FastAPI + Uvicorn | Async REST API, port 7860 (Docker) / 8000 (local) |
| **Evaluation** | RAGAS ≥ 0.2.0 | Faithfulness, Context Recall |
| **Orchestration** | LangChain | Chain, PromptTemplate, Structured Output |
| **Doc Loading** | LlamaIndex + LangChain loaders | PDF, DOCX, TXT, CSV, Markdown |
| **Deployment** | Docker (Python 3.12-slim) | Render.com / HuggingFace Spaces |

---

## Cài đặt & Sử dụng

### 1. Chuẩn bị môi trường

```bash
pip install -r requirements.txt
```

Tạo file `.env` ở thư mục gốc:

```env
# Bắt buộc
GROQ_API_KEY=your_groq_api_key_here

# Tùy chọn — nếu không cấu hình, hệ thống dùng RAM fallback
SUPABASE_DATABASE_URL=postgresql+psycopg2://user:password@host:port/dbname
```

---

### 2. Nạp tài liệu vào hệ thống (Ingestion)

Đặt tài liệu (`.pdf`, `.docx`, `.txt`, `.csv`, `.md`) vào thư mục `data/`:

```bash
python app/service/ingest.py
```

Hệ thống sẽ:
1. Đọc tất cả tài liệu trong `data/` (đệ quy)
2. Chia bằng Semantic Chunker
3. Embed và lưu vào Qdrant tại `VectorDB/`

---

### 3. Khởi chạy Chatbot

**Dùng script menu tương tác (Khuyên dùng):**
```bash
bash run.sh
```

**Chạy Web UI (FastAPI):**
```bash
python api.py
```

**Chạy CLI:**
```bash
python main.py
```

**Test nhanh pipeline:**
```bash
python scripts/test_rag.py
```

**Xóa Vector DB:**
```bash
python scripts/clear_db.py
```

---

### 4. Đánh giá hệ thống (RAGAS Evaluation)

Pipeline đánh giá sử dụng bộ câu hỏi `evaluation/optimization_dataset.json` (tiếng Việt) với 2 metrics:

| Metric | Mô tả |
|---|---|
| **Faithfulness** | Câu trả lời có trung thực với context được truy xuất không? |
| **Context Recall** | Context có đủ để trả lời theo ground truth không? |

**Chạy đánh giá:**
```bash
python evaluation/run_eval.py
```

Kết quả lưu tại `evaluation/results/`:
- `ragas_results_latest.json` — điểm từng mẫu
- `ragas_summary_latest.json` — tóm tắt điểm trung bình
- `ragas_metrics_chart.png` — Bar Chart + Radar Chart
- `ragas_heatmap.png` — Heatmap điểm theo từng mẫu

**Kết quả đánh giá mới nhất** *(5 mẫu — 02/06/2026)*:

| Metric | Điểm | Đánh giá |
|---|---|---|
| **Faithfulness** | **0.8095** | ✅ Tốt |
| **Context Recall** | **0.6571** | 🟡 Khá |
| **Overall Score** | **0.7333** | ✅ Tốt |

> **Nhận xét**: Faithfulness cao (0.81) cho thấy câu trả lời bám sát ngữ cảnh, ít hallucination. Context Recall (0.66) còn dư địa cải thiện — retriever đôi khi bỏ sót một số thông tin cần thiết.

<div align="center">
  <img src="evaluation/results/ragas_metrics_chart.png" alt="RAGAS Metrics Chart" width="800">
</div>

---

## Triển khai bằng Docker

```bash
docker build -t rag-chatbot-app .

docker run -d -p 7860:7860 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/VectorDB:/app/VectorDB \
  --name rag-container \
  rag-chatbot-app
```

*Truy cập tại: `http://localhost:7860`*


