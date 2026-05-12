#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' 

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}    MULTIMODAL RAG CHATBOT RUNNER      ${NC}"
echo -e "${BLUE}=======================================${NC}"

if [ ! -f .env ]; then
    echo -e "${RED}[!] Không tìm thấy file .env. Vui lòng tạo file .env và thêm GOOGLE_API_KEY.${NC}"
    exit 1
fi

echo -e "Chọn một tùy chọn:"
echo -e "${GREEN}1)${NC} Ingest Data (Nạp dữ liệu từ thư mục /data vào VectorDB)"
echo -e "${GREEN}2)${NC} Run Web API (Bắt đầu Web Chatbot trên http://localhost:8000)"
echo -e "${GREEN}3)${NC} Clear VectorDB (Xóa dữ liệu cũ để đổi mô hình)"
echo -e "${GREEN}4)${NC} Thoát"

read -p "Nhập lựa chọn của bạn [1-4]: " choice

case $choice in
    1)
        echo -e "${BLUE}[*] Bắt đầu quá trình nạp dữ liệu...${NC}"
        PYTHONPATH=. python3 app/service/ingest.py
        echo -e "${GREEN}[+] Nạp dữ liệu hoàn tất!${NC}"
        ;;
    2)
        if [ ! -d "VectorDB" ]; then
            echo -e "${RED}[!] Cảnh báo: Thư mục VectorDB chưa tồn tại. Bạn nên chạy tùy chọn 1 trước.${NC}"
            read -p "Vẫn tiếp tục chạy chatbot? (y/n): " confirm
            if [[ ! $confirm =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        echo -e "${BLUE}[*] Đang khởi động Web Server...${NC}"
        PYTHONPATH=. python3 api.py
        ;;
    3)
        echo -e "${RED}[!] Cảnh báo: Tất cả dữ liệu vector sẽ bị xóa.${NC}"
        PYTHONPATH=. python3 scripts/clear_db.py
        ;;
    4)
        echo -e "Tạm biệt!"
        exit 0
        ;;
    *)
        echo -e "${RED}[!] Lựa chọn không hợp lệ.${NC}"
        exit 1
        ;;
esac
