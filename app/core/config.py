import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data")
QDRANT_PATH = os.path.join(BASE_DIR, "VectorDB")

# Collection Names
COLLECTION_NAME = "rag_input"
# If the router uses specific categories, we should define them or handle them
CATEGORIES = ["ky_thuat", "doanh_nghiep", "chung"]

# Model Names
# Sử dụng BERT
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-2.5-flash"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def get_llm(temperature=0.2):
    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=temperature,
        google_api_key=api_key
    )


# Shared Instances
shared_llm = get_llm()