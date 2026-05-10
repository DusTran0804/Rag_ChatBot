# Sử dụng base image Python tối ưu (nhẹ gọn)
FROM python:3.12-slim

# Ngăn chặn Python sinh ra file .pyc
ENV PYTHONDONTWRITEBYTECODE 1
# Ghi log trực tiếp lên console không qua buffer
ENV PYTHONUNBUFFERED 1

# Tạo thư mục làm việc trong container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết (nếu cần thiết cho compiler của một số package)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker Cache
COPY requirements.txt .

# Cài đặt thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code vào container
COPY . .

# Tạo sẵn thư mục data và VectorDB để tránh lỗi quyền ghi (Permissions)
RUN mkdir -p /app/data /app/VectorDB

# Expose cổng 8000 (cổng FastAPI chạy)
EXPOSE 8000

# Lệnh khởi chạy server khi container start
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
