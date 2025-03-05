import os
import requests
import fitz
import uuid
import streamlit as st
from tqdm import tqdm
from dotenv import load_dotenv
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from pinecone_index import initialize_pinecone_index, upsert_vectors
from openai import OpenAI

# Load environment variables
def load_environment():
    """Loads environment variables."""
    load_dotenv()
    return {
        "pinecone_api_key": os.getenv("pinecone_api_key"),
        "host_url": os.getenv("PINECONE_INDEX_HOST"),
        "openai_api_key": os.getenv("OPENAI_API_KEY")
    }

# Initialize OpenAI Client
def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)

# Load Sentence Transformer model
def initialize_embedding_model(device="cpu", model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name, device=device)
    return model, model.get_sentence_embedding_dimension()

# Check if vectors exist in Pinecone
def namespace_has_vectors(index, namespace, embedding_dim):
    try:
        response = index.query(
            namespace=namespace,
            vector=[0] * embedding_dim,
            top_k=1,
            include_values=False
        )
        return len(response.get("matches", [])) > 0
    except Exception as e:
        st.error(f"Error checking namespace: {e}")
        return False

# Download PDF if not exists
def download_pdf(pdf_path, pdf_url):
    if not os.path.exists(pdf_path):
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(pdf_path, "wb") as file:
                file.write(response.content)

# Read and preprocess PDF
def process_pdf(pdf_path):
    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    
    for page_number, page in tqdm(enumerate(doc), desc="Processing PDF"):
        text = page.get_text().replace("\n", " ").strip()
        sentences = [str(sentence) for sentence in list(nlp(text).sents)]
        sentence_chunks = ["".join(sentences[i:i + 10]).strip() for i in range(0, len(sentences), 10)]
        
        pages_and_texts.append({
            "page_number": page_number,
            "text": text,
            "sentence_chunks": sentence_chunks
        })
    
    return pages_and_texts

# Prepare Pinecone vectors
def prepare_pinecone_vectors(pages_and_texts, embedding_model):
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

# Retrieve relevant context from Pinecone
def retrieve_relevant_context(index, query, embedding_model, top_k=3, namespace="nutrition-text"):
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
    query_result = index.query(
        namespace=namespace,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [match["metadata"]["sentences"] for match in query_result["matches"]]

# Format prompt for GPT
def format_prompt(query, context_items):
    context_text = "\n- " + "\n- ".join(context_items)
    return f"""Please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following context items to answer the user query:
{context_text}

User query: {query}
Answer:"""

# Generate AI response using GPT-4o-mini
def generate_answer(client, index, query, embedding_model, top_k=3, namespace="nutrition-text"):
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

# Streamlit UI
def main():
    st.title("📚 AI-Powered RAG: Nutrition Text Assistant")
    st.markdown("Ask me anything about human nutrition, and I'll retrieve relevant knowledge from a scientific textbook!")

    # Load environment variables
    env_vars = load_environment()
    client = initialize_openai_client(env_vars["openai_api_key"])
    embedding_model, embedding_dim = initialize_embedding_model()
    index = initialize_pinecone_index()

    # Check Pinecone vector availability
    with st.spinner("Checking vector database..."):
        if not namespace_has_vectors(index, "nutrition-text", embedding_dim):
            st.warning("No existing vectors found. Processing PDF now...")
            
            # Process PDF and insert into Pinecone
            download_pdf("human-nutrition-text.pdf", "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf")
            pages_and_texts = process_pdf("human-nutrition-text.pdf")
            vectors = prepare_pinecone_vectors(pages_and_texts, embedding_model)
            upsert_vectors(index, vectors, namespace="nutrition-text")
            
            st.success("Vector database successfully updated! You can now ask questions.")

    # User input
    query = st.text_input("🔍 Ask a question about nutrition:")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                response = generate_answer(client, index, query, embedding_model)
                st.markdown("### 🤖 AI Response")
                st.write(response)
        else:
            st.warning("Please enter a query before clicking the button.")

# Run the Streamlit app
if __name__ == "__main__":
    main()