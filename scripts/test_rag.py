import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.service.agent import run_advanced_rag_agent

def main():
    query = "quy hoach tuyen tinh"
    session_id = "test-session-123"
    
    print(f"--- ĐANG KIỂM TRA QUERY: '{query}' ---")
    try:
        answer = run_advanced_rag_agent(query, session_id)
        print("\n--- KẾT QUẢ ---")
        print(answer)
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
