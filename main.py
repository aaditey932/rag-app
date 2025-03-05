import os
import requests
import fitz
import uuid
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load environment variables
def load_environment():
    """Loads environment variables."""
    load_dotenv()
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }

# Initialize OpenAI Client
def initialize_openai_client(api_key):
    """Initializes OpenAI client."""
    return OpenAI(api_key=api_key)

# Initialize Embedding Model
def initialize_embedding_model(device="cpu", model_name="all-mpnet-base-v2"):
    """Loads the embedding model."""
    model = SentenceTransformer(model_name, device=device)
    return model

# Download PDF if not available
def download_pdf(pdf_path, pdf_url):
    """Downloads the PDF if it does not exist."""
    if not os.path.exists(pdf_path):
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(pdf_path, "wb") as file:
                file.write(response.content)

# Process PDF and Split into Chunks
def process_pdf(pdf_path):
    """Reads and processes PDF into structured text."""
    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    
    for page_number, page in tqdm(enumerate(doc), desc="Processing PDF"):
        text = page.get_text().replace("\n", " ").strip()
        sentences = [str(sentence) for sentence in list(nlp(text).sents)]
        sentence_chunks = [" ".join(sentences[i:i+10]).strip() for i in range(0, len(sentences), 10)]
        
        pages_and_texts.append({
            "page_number": page_number,
            "text": text,
            "sentences": sentences,
            "sentence_chunks": sentence_chunks
        })
    
    return pages_and_texts

# Convert Chunks to Embeddings
def prepare_vectors(pages_and_texts, embedding_model):
    """Prepares text chunks as vectors for similarity search."""
    vectors = []
    for item in pages_and_texts:
        for chunk in item["sentence_chunks"]:
            embedding = embedding_model.encode(chunk, normalize_embeddings=True).tolist()
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "page_number": item["page_number"],
                    "sentences": chunk
                }
            })
    return vectors

# Custom Cosine Similarity Search (Without Pinecone)
def custom_similarity_search(query_text, stored_vectors, embedding_model, top_k=3):
    """Performs similarity search manually using cosine similarity."""
    try:
        # Convert query to embedding vector
        query_embedding = embedding_model.encode(query_text, normalize_embeddings=True)

        # Compute cosine similarity
        similarities = []
        for item in stored_vectors:
            stored_embedding = np.array(item["values"])
            similarity = np.dot(query_embedding, stored_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding))
            similarities.append((item, similarity))

        # Sort by similarity (highest first)
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        # Retrieve top-k results
        top_results = [
            {
                "id": item["id"],
                "score": score,
                "page_number": item["metadata"].get("page_number", "Unknown"),
                "sentences": item["metadata"].get("sentences", "")
            }
            for item, score in similarities[:top_k]
        ]

        return top_results

    except Exception as e:
        print(f"Error performing similarity search: {e}")
        return []

# Format Prompt for OpenAI LLM
def format_prompt(query, context_items):
    """Formats the query and retrieved context into a structured prompt for OpenAI."""
    context_text = "\n- " + "\n- ".join(context_items)
    return f"""Please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following context items to answer the user query:
{context_text}

User query: {query}
Answer:"""

# Generate Answer Using OpenAI LLM
def generate_answer(client, stored_vectors, query, embedding_model, top_k=3):
    """Retrieves relevant context and generates response using OpenAI's GPT-4o-mini."""
    context_items = custom_similarity_search(query, stored_vectors, embedding_model, top_k)
    
    if not context_items:
        return "No relevant information found in the knowledge base."

    context_text = [item["sentences"] for item in context_items]
    prompt = format_prompt(query, context_text)

    # Generate response using OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=256,
        top_p=0.95
    )
    return completion.choices[0].message.content

# Main Execution Function
def main():
    """Main execution function."""
    env_vars = load_environment()
    client = initialize_openai_client(env_vars["openai_api_key"])
    embedding_model = initialize_embedding_model()

    # Download and Process PDF
    pdf_path = "human-nutrition-text.pdf"
    pdf_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    download_pdf(pdf_path, pdf_url)
    pages_and_texts = process_pdf(pdf_path)

    # Prepare Vectors for Similarity Search
    stored_vectors = prepare_vectors(pages_and_texts, embedding_model)

    # Query and Get Response
    query = input("Hello! How can I help you today?\n> ")
    response = generate_answer(client, stored_vectors, query, embedding_model)
    
    print(response)

# Run the Script
if __name__ == "__main__":
    main()
