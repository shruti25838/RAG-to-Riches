from sentence_transformers import SentenceTransformer
import numpy as np

# Load MiniLM model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> list:
    """
    Convert input text into float32 embedding vector using MiniLM.
    """
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype(np.float32).tolist()
