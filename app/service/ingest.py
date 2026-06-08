import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)

import hashlib
import json

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

def get_file_hash(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception:
        return None

def load_document():
    langchain_docs = []
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return None

    checkpoint_path = os.path.join(QDRANT_PATH, "ingest_checkpoint.json")
    checkpoint = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
        except Exception:
            checkpoint = {}

    processed_files = {}
    new_files_count = 0

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in LOADER_MAPPING:
                file_path = os.path.join(root, file)
                
                file_hash = get_file_hash(file_path)
                if file_hash is None:
                    continue
                
                if file_path in checkpoint and checkpoint[file_path] == file_hash:
                    processed_files[file_path] = file_hash
                    continue

                try:
                    loader_cls = LOADER_MAPPING[ext]
                    loader = loader_cls(file_path)
                    langchain_docs.extend(loader.load())
                    processed_files[file_path] = file_hash
                    new_files_count += 1
                    print(f"Loaded new/updated file: {file}")
                except Exception as e:
                    print(f"[Error] Failed to load {file}: {e}")
            else:
                if not file.startswith('.'):
                    print(f"Ignore format : {file}")

    if new_files_count == 0:
        print("No new documents to ingest. Checkpoint is up to date.")
        return None

    if not langchain_docs:
        return None

    from langchain_experimental.text_splitter import SemanticChunker
    
    embeddings = get_embeddings()
    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_amount=0.8
    )
    chunks = semantic_chunker.split_documents(langchain_docs)

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

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        vector_name="dense"
    )
    vector_store.add_documents(chunks)

    try:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, ensure_ascii=False, indent=4)
        print(f"Checkpoint saved successfully with {len(processed_files)} files.")
    except Exception as e:
        print(f"[Error] Failed to save checkpoint: {e}")

    return vector_store

if __name__ == "__main__":
    load_document()