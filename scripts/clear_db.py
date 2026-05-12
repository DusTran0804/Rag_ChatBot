import shutil
import os
from app.core.config import QDRANT_PATH, COLLECTION_NAME, CATEGORIES
from qdrant_client import QdrantClient

def clear_vector_db():
    print(f"Checking Qdrant database at: {QDRANT_PATH}")
    
    # Method 1: Delete via Client (Cleaner if Qdrant is running or using local path)
    try:
        client = QdrantClient(path=QDRANT_PATH)
        all_collections = CATEGORIES + [COLLECTION_NAME]
        
        for coll in all_collections:
            if client.collection_exists(coll):
                print(f"Deleting collection: {coll}")
                client.delete_collection(coll)
            else:
                print(f"Collection {coll} does not exist.")
    except Exception as e:
        print(f"Error deleting collections via client: {e}")
        
    # Method 2: Force delete the storage directory if client method is not enough
    if os.path.exists(QDRANT_PATH):
        print(f"Removing storage directory: {QDRANT_PATH}")
        try:
            shutil.rmtree(QDRANT_PATH)
            print("Successfully removed VectorDB directory.")
        except Exception as e:
            print(f"Error removing directory: {e}")
    else:
        print("VectorDB directory already cleared.")

if __name__ == "__main__":
    confirm = input("This will delete all existing vector data. Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        clear_vector_db()
    else:
        print("Operation cancelled.")
