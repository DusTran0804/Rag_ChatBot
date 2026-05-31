import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import DATA_PATH, QDRANT_PATH, COLLECTION_NAME, get_embeddings, CATEGORIES

LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".md": UnstructuredMarkdownLoader,
}

def load_document():
    langchain_docs = []
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return None

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in LOADER_MAPPING:
                file_path = os.path.join(root, file)
                try:
                    loader_cls = LOADER_MAPPING[ext]
                    loader = loader_cls(file_path)
                    langchain_docs.extend(loader.load())
                except Exception as e:
                    print(f"[Error")
            else:
                if not file.startswith('.'):
                    print(f"Ignore format : {file}")

    if not langchain_docs:
        return None

    from langchain_experimental.text_splitter import SemanticChunker
    
    embeddings = get_embeddings()
    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_amount=0.8
    )
    chunks = semantic_chunker.split_documents(langchain_docs)

    # 3. Vector Store setup
    client = QdrantClient(path=QDRANT_PATH)
    all_collections = CATEGORIES + [COLLECTION_NAME]
    embed_dim = len(embeddings.embed_query("test"))
    
    for coll in all_collections:
        if not client.collection_exists(coll):
            client.create_collection(
                collection_name=coll,
                vectors_config={
                    "dense": models.VectorParams(size=embed_dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )

    # 4. Ingest into default collection
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        vector_name="dense"
    )
    vector_store.add_documents(chunks)
    return vector_store

if __name__ == "__main__":
    load_document()