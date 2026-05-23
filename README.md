# Advanced Enterprise RAG Chatbot

Hệ thống Chatbot RAG (Retrieval-Augmented Generation) thông minh, được thiết kế để xử lý đa luồng câu hỏi, hỗ trợ giao diện Web trực quan và tích hợp sâu với các mô hình ngôn ngữ tiên tiến của Google và HuggingFace.

## Sơ đồ Kiến trúc Hệ thống

Dưới đây là sơ đồ luồng hoạt động (Workflow) của hệ thống, bao gồm quá trình Nạp dữ liệu (Offline Ingestion) và quá trình Truy vấn thời gian thực (Online Querying):

```mermaid
flowchart TD
    %% Ingestion Pipeline
    subgraph Ingestion [Offline Data Ingestion]
        direction LR
        A1((Actor)) --> B1[Load Document]
        B1 --> C1[Chunking Document]
        C1 --> D1[Embedding Model]
    end

    D1 -- Save to Qdrant --> VDB[(Vector DB)]

    %% Query Pipeline
    U1((Client)) --> Q1([Query])
    
    Q1 --> QT[Query Transform]
    QT --> QR[Query Router]
    
    %% Routing
    QR -- question from documents --> R1[question from documents]
    QR -- Real-time question --> R2[Real-time question]
    QR -- Other Question --> R3[Other Question]
    
    %% Branch 1: RAG Documents
    R1 --> D2[Documents]
    VDB -. Retrieval .-> D2
    D2 --> RR[Re-rank and filter]
    RR --> LLM[LLM]
    
    %% Branch 2: Real-time Search Agent
    R2 --> AG[AI Agent]
    AG --> D3[Documents]
    D3 --> LLM
    
    %% Branch 3: Direct Answer
    R3 --> ANS([Answer])
    LLM --> ANS
```

## Frameworks sử dụng

Dự án này được xây dựng trên một stack công nghệ hiện đại và chuyên nghiệp:

### 1. Xử lý Logic & Orchestration
- **[LangChain](https://python.langchain.com/)**: Bộ khung lõi (Core Framework) điều phối toàn bộ luồng RAG, khởi tạo LLM, kết nối VectorDB và thiết kế Agent thông minh.
- **[LlamaIndex](https://www.llamaindex.ai/)**: Sử dụng `SimpleDirectoryReader` chuyên biệt để đọc, bóc tách và tiền xử lý nhiều định dạng tài liệu phức tạp (PDF, Text, Hình ảnh đa phương thức).
- **Query Transform:** Multi-step/Sub-query Decomposition & **Query Routing:** Logical Router.
- **Memory Contextualization (Query Rewrite)**: Hệ thống duy trì lịch sử hội thoại (Chat History) và tự động sử dụng LLM để viết lại các câu hỏi mang tính chất nối tiếp (ví dụ: chứa từ "nó", "thế còn") thành một câu hỏi độc lập và đầy đủ ngữ nghĩa trước khi đem đi xử lý, giúp bot luôn hiểu đúng bối cảnh giao tiếp.
- **Retrieval & Re-ranking Pipeline**: Kết hợp tìm kiếm đa luồng (truy xuất các tài liệu cho toàn bộ các sub-queries từ Qdrant) cùng với cơ chế deduplication (loại bỏ trùng lặp). Sau đó, áp dụng kỹ thuật **Re-ranking** (chấm điểm lại bằng mô hình Cross-Encoder) để đánh giá chéo và chọn lọc ra những ngữ cảnh (context) có độ chính xác và tương đồng cao nhất với ý định gốc của người dùng trước khi đưa vào LLM.
### 2. Mô hình AI (Models)
- **Groq LLaMA 3.3 (`llama-3.3-70b-versatile`)**: Làm bộ não chính (LLM) để tổng hợp, suy luận và tạo ra câu trả lời cuối cùng, tối ưu cho tốc độ siêu tốc và khả năng hiểu ngữ cảnh vượt trội thông qua API của Groq.
- **HuggingFace Sentence-Transformers**: 
   - **Embedding**: Sử dụng `bkai-foundation-models/vietnamese-bi-encoder` (mô hình Bi-Encoder tối ưu cho tiếng Việt) để chuyển đổi văn bản thành vector (768 chiều).
  - **Re-ranking**: Sử dụng Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) để chấm điểm và sắp xếp lại mức độ liên quan của các tài liệu sau khi Retrieval, tăng tính chính xác tuyệt đối.

### 3. Cơ sở dữ liệu Vector (Vector Database)
- **[Qdrant](https://qdrant.tech/)**: Vector DB siêu tốc độ để lưu trữ, index và tìm kiếm các Embeddings. Chạy trực tiếp ở chế độ Local (lưu thư mục `VectorDB`).

### 4. Backend Server & Web UI
- **[FastAPI](https://fastapi.tiangolo.com/) & Uvicorn**: Xây dựng máy chủ RESTful API bất đồng bộ siêu nhanh, mạnh mẽ, phục vụ trực tiếp Web UI.
- **Pydantic**: Ràng buộc dữ liệu (Data Validation) chặt chẽ cho Request/Response.
- **HTML / CSS / Vanilla JS**: Giao diện người dùng Web Chat trực quan, mượt mà (hỗ trợ Dark Mode, render Markdown).

### 5. Triển khai (Deployment)
- **[Docker](https://www.docker.com/)**: Đóng gói toàn bộ ứng dụng (containerized) thành Image thông qua `Dockerfile`, đảm bảo tính ổn định và tính chuyên nghiệp khi mang lên môi trường Production.

---

## Cài đặt & Sử dụng (Môi trường Local)

### 1. Chuẩn bị
Yêu cầu: `Python 3.10+`.
Bản sao dự án về và cài đặt thư viện:
```bash
pip install -r requirements.txt
```

Cấu hình API Key: Tạo file `.env` ở thư mục gốc:
```env
GROQ_API_KEY=your_groq_api_key_here
```
### Công cụ tự động (Khuyên dùng)
Dự án có sẵn script `run.sh` giúp bạn dễ dàng quản lý hệ thống qua giao diện menu tương tác mà không cần nhớ lệnh Python:
```bash
bash run.sh
```
*Giao diện Menu sẽ cho phép bạn: (1) Nạp dữ liệu, (2) Chạy Web Chatbot, (3) Xóa trắng Database.*

---
*(Hoặc bạn có thể chạy thủ công từng bước như bên dưới)*

### 2. Nạp tài liệu vào hệ thống (Ingestion)
Bỏ các file dữ liệu (.pdf, .txt...) vào thư mục `data/` rồi chạy lệnh:
```bash
python app/service/ingest.py
```
*(Hệ thống sẽ tiến hành đọc, chunking, nhúng (embedding) và lưu vào Qdrant tại thư mục `VectorDB/`)*

### 3. Khởi chạy Web Chatbot
Bạn có thể chạy trực tiếp bằng Terminal CLI:
```bash
python main.py
```
Hoặc bật **Web UI** bằng FastAPI:
```bash
python api.py
# Truy cập: http://localhost:8000
```

### 4. Đánh giá hệ thống (Ragas Evaluation)

Hệ thống được tích hợp sẵn công cụ đánh giá tự động bằng framework **Ragas** để đo lường độ hiệu quả của pipeline RAG qua các chỉ số:
- **Faithfulness**: Độ trung thực của câu trả lời so với ngữ cảnh.
- **Answer Relevancy**: Mức độ bám sát câu hỏi.
- **Context Precision & Recall**: Độ chính xác và đầy đủ của tài liệu truy xuất.
- **Answer Correctness**: Độ chính xác tổng thể so với đáp án chuẩn.

Để tiến hành đánh giá, hãy đảm bảo bạn đã cung cấp biến `GROQ_API_KEY` trong file `.env` và chạy lệnh:
```bash
python evaluation/run_eval.py
```
*(Kết quả chi tiết và các biểu đồ trực quan như Radar Chart, Heatmap sẽ được lưu vào thư mục `evaluation/results/`)*

<div align="center">
  <img src="evaluation/results/ragas_metrics_chart.png" alt="Ragas Metrics Chart" width="800">
</div>

---

## Triển khai bằng Docker (Production)

Để chạy dự án một cách chuyên nghiệp và độc lập, hãy sử dụng Docker.

**Build Image:**
```bash
docker build -t rag-chatbot-app .
```

**Run Container:**
```bash
docker run -d -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/VectorDB:/app/VectorDB \
  --name rag-container \
  rag-chatbot-app
```
*Truy cập UI Chatbot tại: `http://localhost:8000`*
