import os
import requests
import fitz
import torch
import uuid
from tqdm import tqdm
from dotenv import load_dotenv
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from pinecone_index import initialize_pinecone_index, upsert_vectors  # Pinecone client functions
from openai import OpenAI

def load_environment():
    """Loads environment variables."""
    load_dotenv()
    return {
        "pinecone_api_key": os.getenv("pinecone_api_key"),
        "host_url": os.getenv("PINECONE_INDEX_HOST"),
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }

def initialize_openai_client(api_key):
    """Initializes OpenAI client."""
    return OpenAI(api_key=api_key)

def initialize_embedding_model(device="cuda", model_name="all-mpnet-base-v2"):
    """Loads the embedding model."""
    model = SentenceTransformer(model_name, device=device)
    return model, model.get_sentence_embedding_dimension()

def namespace_has_vectors(index, namespace, embedding_dim):
    """Checks if vectors exist in a Pinecone namespace."""
    try:
        response = index.query(
            namespace=namespace,
            vector=[0] * embedding_dim,
            top_k=1,
            include_values=False
        )
        return len(response.get("matches", [])) > 0
    except Exception as e:
        print(f"Error checking namespace: {e}")
        return False

def download_pdf(pdf_path, pdf_url):
    """Downloads the PDF if it does not exist."""
    if not os.path.exists(pdf_path):
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(pdf_path, "wb") as file:
                file.write(response.content)

def process_pdf(pdf_path):
    """Reads and processes PDF into structured text."""
    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc), desc="Processing PDF"):
        text = page.get_text().replace("\n", " ").strip()
        pages_and_texts.append({
            "page_number": page_number,
            "text": text,
            "sentences": [str(sentence) for sentence in list(nlp(text).sents)],
        })
        pages_and_texts[-1]["sentence_chunks"] = [
            "".join(pages_and_texts[-1]["sentences"][i:i + 10]).strip()
            for i in range(0, len(pages_and_texts[-1]["sentences"], 10))
        ]
    return pages_and_texts

def prepare_pinecone_vectors(pages_and_texts, embedding_model):
    """Prepares vectors for Pinecone indexing."""
    vectors = []
    for item in pages_and_texts:
        for chunk in item["sentence_chunks"]:
            embedding = embedding_model.encode(chunk, normalize_embeddings=True).tolist()
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "page_number": item["page_number"],
                    "source": "human-nutrition-text.pdf",
                    "sentences": chunk
                }
            })
    return vectors

def retrieve_relevant_context(index, query, embedding_model, top_k=3, namespace="nutrition-text"):
    """Retrieves relevant context from Pinecone."""
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
    query_result = index.query(
        namespace=namespace,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [match["metadata"]["sentences"] for match in query_result["matches"]]

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

def generate_answer(client, index, query, embedding_model, top_k=3, namespace="nutrition-text"):
    """Retrieves context and generates a response using OpenAI's GPT-4o-mini."""
    context_items = retrieve_relevant_context(index, query, embedding_model, top_k, namespace)
    if not context_items:
        return "No relevant information found in the knowledge base."
    prompt = format_prompt(query, context_items)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=256,
        top_p=0.95
    )
    return completion.choices[0].message.content

def main():
    """Main execution function."""
    env_vars = load_environment()
    client = initialize_openai_client(env_vars["openai_api_key"])
    embedding_model, embedding_dim = initialize_embedding_model()
    index = initialize_pinecone_index()
    
    if not namespace_has_vectors(index, "nutrition-text", embedding_dim):
        download_pdf("human-nutrition-text.pdf", "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf")
        pages_and_texts = process_pdf("human-nutrition-text.pdf")
        vectors = prepare_pinecone_vectors(pages_and_texts, embedding_model)
        upsert_vectors(index, vectors, namespace="nutrition-text")
    
    query = input("Hello! How can I help you today?")
    response = generate_answer(client, index, query, embedding_model)
    print(response)

if __name__ == "__main__":
    main()
