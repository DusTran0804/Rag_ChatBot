import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data")
QDRANT_PATH = os.path.join(BASE_DIR, "VectorDB")

COLLECTION_NAME = "rag_input"
CATEGORIES = ["ky_thuat", "doanh_nghiep", "chung"]
SUPABASE_DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
if SUPABASE_DATABASE_URL:
    SUPABASE_DATABASE_URL = SUPABASE_DATABASE_URL.strip().replace('\r', '').replace('\n', '')

EMBED_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
RERANK_MODEL_NAME = "amberoad/bert-multilingual-passage-reranking-msmarco"

def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def get_llm(temperature=0.2):
    api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=temperature,
        groq_api_key=api_key,
        max_retries=10,
        timeout=60.0
    )

shared_llm = get_llm()

def invoke_chain_with_retry(chain, input_data, max_retries=10, initial_delay=5.0):
    import time
    import logging
    logger = logging.getLogger(__name__)
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            err_msg = str(e)
            if attempt == max_retries:
                logger.error(f"Failed to invoke chain after {max_retries} attempts: {e}")
                raise e
            logger.warning(f"[Attempt {attempt}/{max_retries}] LLM invoke error: {err_msg}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)