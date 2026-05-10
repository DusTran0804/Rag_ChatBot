import os
from llama_index.core import SimpleDirectoryReader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import DATA_PATH, QDRANT_PATH, COLLECTION_NAME, get_embeddings, CATEGORIES

def load_document():
    # 1. Load documents using LlamaIndex
    llama_docs = SimpleDirectoryReader(
        input_dir=DATA_PATH,
        recursive=True
    ).load_data()

    # 2. Convert to LangChain documents
    langchain_docs = []
    for doc in llama_docs:
        lc_doc = Document(page_content=doc.text, metadata=doc.metadata)
        langchain_docs.append(lc_doc)

    if not langchain_docs:
        return None

    # 3. Chunking
    embeddings = get_embeddings()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(langchain_docs)

    # 4. Vector Store setup
    client = QdrantClient(path=QDRANT_PATH)
    
    # Initialize collections for all categories + default
    all_collections = CATEGORIES + [COLLECTION_NAME]
    
    for coll in all_collections:
        if not client.collection_exists(coll):
            client.create_collection(
                collection_name=coll,
                vectors_config={
                    "dense": models.VectorParams(size=len(embeddings.embed_query("test")), distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )

    # 5. Ingest into default collection
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        vector_name="dense"
    )
    vector_store.add_documents(chunks)
    return vector_store

if __name__ == "__main__":
    os.makedirs(DATA_PATH, exist_ok=True)
    load_document()