import chromadb
import os

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name=os.getenv("CHROMA_COLLECTION_NAME")
)


def ingest_documents(docs):
    """Ingest documents into ChromaDB using 'all-MiniLM-L6-v2' Sentence Transformer

    Args:
        docs: list of strings (document chunks)
    """
    # Ids for the docs
    ids = [f"chunk_{i}" for i in range(len(docs))]

    # Ingest chunks into the collection
    collection.add(documents=docs, ids=ids)

    return len(docs)


def query_documents(query_text, n_results=3):
    """Query the collection for relevant documents

    Args:
        query_text: string to search for
        n_results: number of results to return

    Returns:
        List of relevant document chunks
    """
    results = collection.query(query_texts=[query_text], n_results=n_results)
    if 'documents' in results and results['documents']:
        return results['documents'][0]
    else:
        return []