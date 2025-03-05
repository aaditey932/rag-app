# 📚 AI-Powered RAG: Nutrition Text Assistant

## 🚀 Overview
This project implements an **AI-powered Retrieval-Augmented Generation (RAG) system** designed to extract relevant information from **scientific research papers** and generate insightful responses. Using **Pinecone, OpenAI's GPT-4o, and Streamlit**, the system enhances accessibility to scientific knowledge in **human nutrition**.

---

## 🎯 Objective
- Enable **AI-driven semantic search** and **response generation** for research papers.
- Utilize **Retrieval-Augmented Generation (RAG)** to enhance factual accuracy and coherence.
- Improve **response faithfulness** using context-aware retrieval techniques.

---

## 🛠️ Tech Stack
- **OpenAI GPT-4o-mini** – AI-powered response generation
- **Pinecone** – Scalable vector search for retrieval
- **SentenceTransformers (all-mpnet-base-v2)** – Embedding model for text chunking
- **Streamlit** – Interactive UI for user interaction
- **Ragas Evaluation** – Automated performance assessment

---

## 🔍 How It Works
1. **Text Ingestion & Processing:** Extracts content from research papers (PDFs) and splits it into meaningful chunks.
2. **Embedding Generation:** Converts text into dense vectors using **SentenceTransformers**.
3. **Vector Storage & Retrieval:** Stores embeddings in **Pinecone** and retrieves relevant text upon query.
4. **Response Generation:** Uses **GPT-4o** to generate AI responses based on retrieved context.
5. **Evaluation Metrics:** Measures performance with **context recall, faithfulness, and factual correctness**.

---

## 📊 Performance Evaluation
| Metric                 | Score  |
|------------------------|--------|
| **Context Recall**     | 86.67% |
| **Faithfulness**       | 61.83% |
| **Factual Correctness** | 67.00% |

### 🔹 Key Takeaways:
- **Strong retrieval accuracy** ensures relevant document extraction.
- **Response faithfulness** needs improvement for more aligned answers.
- **Future Enhancements:** Reranking techniques & fine-tuned models for factual correctness.

---

## 📥 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/AI-Nutrition-RAG.git
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables
Create a **.env** file and add the following:
```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_HOST=your_pinecone_host
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 🏗️ Project Structure
```
📂 AI-Nutrition-RAG
├── 📄 main.py            # Main execution file
├── 📄 streamlit.py       # Streamlit UI interface
├── 📄 pinecone_index.py  # Pinecone vector storage logic
├── 📄 rag_evaluation.py  # RAG evaluation metrics
├── 📄 requirements.txt   # Dependencies
├── 📄 README.md          # Project documentation
└── 📂 data               # Research papers (PDFs)
```

---

## 🛠️ Future Improvements
✅ **Improve response faithfulness** using reranking models (Cross-Encoder reranking).  
✅ **Expand dataset** with more domain-specific research papers.  
✅ **Optimize retrieval with hybrid search** (dense + sparse retrieval).  
✅ **Automate evaluation benchmarks** using external datasets.

---

## 📌 References
- [OpenAI API](https://platform.openai.com/)
- [Pinecone](https://www.pinecone.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io/)

---

## 🤝 Contributing
Pull requests are welcome! Feel free to submit improvements or raise issues.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🎥 Demo Video
🎬 **Watch the full demo here:** [YouTube Video](https://youtu.be/1J63noZbh68)

For any queries, feel free to reach out! 🚀
