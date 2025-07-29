import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ.get("PINECONE_INDEX_NAME", "rag-to-riches")
namespace = os.environ.get("PINECONE_NAMESPACE", "resume")

index = pc.Index(index_name)

def upsert_embedding(id, embedding, metadata):
    index.upsert(
        vectors=[{
            "id": id,
            "values": embedding,
            "metadata": metadata
        }],
        namespace=namespace
    )

def query_embedding(query_vector, top_k=3):
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    return result.get("matches", [])

def delete_all_chunks():
    try:
        print(f"🧹 Deleting all vectors from namespace '{namespace}'")
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        print(f"⚠️ Warning: Failed to delete namespace '{namespace}'. Error: {e}")

# Export for external use
__all__ = ['upsert_embedding', 'query_embedding', 'delete_all_chunks', 'index', 'namespace']
