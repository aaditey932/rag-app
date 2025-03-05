import os
import math
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load API Key
load_dotenv()
pinecone_api_key = os.getenv("pinecone_api_key")

# Initialize Pinecone client
pc = Pinecone(pinecone_api_key)

# Define Index Name & Model
INDEX_NAME = "rag-index"
EMBEDDING_MODEL = "all-mpnet-base-v2"
DEVICE = "cpu"
BATCH_SIZE = 100  # Adjust if needed

# Load embedding model to get dimensions
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

def initialize_pinecone_index():
    """
    Creates or retrieves an existing Pinecone index and returns the index object.
    """
    existing_indexes = [index["name"] for index in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,  # Model dimension
            metric="cosine",  # Metric for similarity search
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    # Get index details and print the host URL
    index_info = pc.describe_index(INDEX_NAME)
    host_url = index_info['host']
    os.environ["PINECONE_INDEX_HOST"] = host_url

    print(f"Pinecone index host: {host_url}")

    # Connect to the index
    return pc.Index(INDEX_NAME, host=host_url)

def namespace_exists(index, namespace):
    """
    Checks if a namespace exists by querying for any vector.
    Returns True if vectors exist in the namespace, False otherwise.
    """
    try:
        query_result = index.query(
            namespace=namespace, 
            vector=[0] * EMBEDDING_DIM,  # Dummy vector
            top_k=1,  # Check if at least 1 vector exists
            include_metadata=False
        )
        return len(query_result["matches"]) > 0
    except Exception as e:
        print(f"Error checking namespace '{namespace}': {e}")
        return False

def upsert_vectors(index, vectors, namespace="nutrition-text"):
    """
    Upserts vectors into the Pinecone index in batches, skipping if namespace already exists.
    """
    if namespace_exists(index, namespace):
        print(f"Namespace '{namespace}' already contains vectors. Skipping upsert.")
        return

    num_batches = math.ceil(len(vectors) / BATCH_SIZE)
    
    for i in range(num_batches):
        batch = vectors[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"Upserted batch {i+1}/{num_batches}")

    print(f"Successfully upserted {len(vectors)} vectors into Pinecone!")