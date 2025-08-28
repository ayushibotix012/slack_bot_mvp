import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import shutil
# Path to save/load FAISS index
FAISS_FOLDER = "faiss_index"

# Global vector store
vector_store = None

# Embedding model
embedding_model = OpenAIEmbeddings()

def load_vector_store():
    """
    Load FAISS vector store from disk if it exists.
    """
    global vector_store
    if os.path.exists(os.path.join(FAISS_FOLDER, "index.faiss")):
        vector_store = FAISS.load_local(
            FAISS_FOLDER, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
    else:
        vector_store = None

def save_vector_store():
    """
    Save FAISS vector store to disk.
    """
    global vector_store
    if vector_store:
        vector_store.save_local(FAISS_FOLDER)

def add_to_vector_store(chunks: list[str]):
    """
    Append new chunks to the existing vector store instead of overwriting.
    """
    global vector_store

    if not chunks:
        return

    # Ensure we have the existing store loaded
    load_vector_store()

    if vector_store:
        # Append to existing index
        vector_store.add_texts(chunks)
    else:
        # Create new index
        vector_store = FAISS.from_texts(chunks, embedding_model)

    save_vector_store()

def query_vector_store(query: str, k: int = 10) -> str:
    """
    Retrieve top-k relevant chunks for a query.
    """
    global vector_store

    if vector_store is None:
        load_vector_store()

    if vector_store is None:
        return "⚠️ Vector store is empty. Please upload documents first."

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])




def clear_vector_store():
    """
    Clear the FAISS vector store both in memory and on disk.
    """
    global vector_store
    vector_store = None  # Reset in-memory store

    if os.path.exists(FAISS_FOLDER):
        try:
            shutil.rmtree(FAISS_FOLDER)  # Delete entire folder
            print("✅ Vector store cleared successfully.")
            return True
        except Exception as e:
            print(f"⚠️ Failed to clear vector store: {e}")
            return False
    else:
        print("ℹ️ No vector store found to clear.")
        return True
