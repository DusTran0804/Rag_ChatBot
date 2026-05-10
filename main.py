import os
from app.service.agent import run_advanced_rag_agent
from app.core.config import get_llm

def main():
    print("--- Multimodal RAG Chatbot ---")
    session_id = "test_session_1"
    
    while True:
        query = input("\nBạn: ")
        if query.lower() in ["exit", "quit", "thoát"]:
            break
            
        try:
            response = run_advanced_rag_agent(query, session_id)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\nLỗi: {e}")

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    main()
