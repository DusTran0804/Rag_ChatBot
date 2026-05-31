import sys
import os
from dotenv import load_dotenv

# Nạp file .env trực tiếp để tránh import các thư viện nặng (như torch/huggingface)
load_dotenv()
SUPABASE_DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
if SUPABASE_DATABASE_URL:
    SUPABASE_DATABASE_URL = SUPABASE_DATABASE_URL.strip().replace('\r', '').replace('\n', '')

def test_connection():
    print("--- KIỂM TRA KẾT NỐI SUPABASE DATABASE ---")
    if not SUPABASE_DATABASE_URL:
        print("[LỖI] Biến môi trường SUPABASE_DATABASE_URL chưa được thiết lập hoặc chưa được nạp.")
        sys.exit(1)
        
    print(f"URL cơ sở dữ liệu: {SUPABASE_DATABASE_URL}")
    print("Đang kết nối tới Supabase...")
    
    try:
        from sqlalchemy import create_engine, text
        
        # Tạo engine kết nối
        engine = create_engine(SUPABASE_DATABASE_URL)
        
        # Thử thực thi câu lệnh SQL đơn giản
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            row = result.fetchone()
            print("[THÀNH CÔNG] Đã kết nối tới PostgreSQL Supabase!")
            print(f"Phiên bản PostgreSQL: {row[0]}")
            
            # Thử nghiệm truy vấn bảng chat_history nếu có
            try:
                table_check = conn.execute(text("SELECT count(*) FROM chat_history;"))
                print(f"[THÔNG TIN] Bảng chat_history đã tồn tại. Số bản ghi hiện tại: {table_check.fetchone()[0]}")
            except Exception as e_table:
                print(f"[THÔNG TIN] Bảng chat_history chưa tồn tại hoặc chưa thể đọc ({e_table}). Nó sẽ tự động được tạo khi chạy chatbot.")
                
    except Exception as e:
        print(f"[LỖI] Kết nối thất bại: {e}")
        print("\nGợi ý khắc phục:")
        print("1. Kiểm tra lại thông tin mật khẩu và tên host trong file .env")
        print("2. Đảm bảo mật khẩu của bạn đã được URL-encode nếu chứa ký tự đặc biệt (ví dụ: @ thành %40)")
        sys.exit(1)

if __name__ == "__main__":
    test_connection()
