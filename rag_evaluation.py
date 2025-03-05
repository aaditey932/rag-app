import os
import streamlit as rag
from dotenv import load_dotenv
from pinecone_index import initialize_pinecone_index
from streamlit import retrieve_relevant_context, initialize_embedding_model
from openai import OpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from langchain.chat_models import ChatOpenAI
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# Load environment variables
load_dotenv()

# Initialize required components
index = initialize_pinecone_index()
embedding_model, embedding_dim = initialize_embedding_model()

env_vars = rag.load_environment()
client = rag.initialize_openai_client(env_vars["openai_api_key"])

# Sample queries and expected responses
sample_queries = [
    "What are the macronutrients essential for human health?",
    "How does vitamin C benefit the immune system?",
    "What are the effects of a high-protein diet on muscle growth?"
]

expected_responses = [
    """The macronutrients essential for human health include:
    Water – Vital for transporting essential nutrients, disposing of waste, regulating body temperature, and enabling chemical reactions necessary for life. More than 60% of the human body is composed of water.
    Protein – Necessary for tissue formation, cell repair, and the production of hormones and enzymes. It plays a crucial role in building strong muscles and maintaining a healthy immune system.
    Carbohydrates – Provide a readily available source of energy for the body and contribute to the structural components needed for cell formation.
    Fats (Lipids) – Serve as stored energy, function as structural components of cells, act as signaling molecules for proper cellular communication, provide insulation to vital organs, and help regulate body temperature.
    These macronutrients are required in large quantities to ensure proper body function, growth, and overall health.""",

    """Vitamin C plays several roles in supporting the immune system. While there is no strong evidence that it prevents colds, studies suggest that it can slightly reduce the severity and duration of cold symptoms. Many people increase their vitamin C intake through diet or supplements when they have a cold, although taking megadoses at the onset of a cold has not been shown to provide additional benefits.
    Additionally, vitamin C contributes to overall immune function, helping the body maintain its defenses against infections and illnesses. However, the best way to obtain its benefits is through a diet rich in fruits and vegetables rather than relying solely on supplements.""",

    """A high-protein diet can have both positive and potential negative effects on muscle growth.
    Effects on Muscle Growth:
    Muscle Maintenance and Growth: Higher protein intake (1.2 to 1.5 grams per kilogram of body weight per day) may help prevent muscle loss, particularly in aging adults. This suggests that sufficient protein is beneficial for maintaining and increasing muscle mass.
    No Added Benefits from Supplements: While many physically active individuals consume protein or amino acid supplements, scientific evidence indicates that these supplements are no more effective than whole food sources of protein when energy intake is adequate.
    Branched-Chain Amino Acids (BCAAs) and Performance: Although BCAAs, such as leucine, are often marketed for muscle growth and athletic performance, most studies have not shown significant performance-enhancing effects.
    Potential Downsides:
    Health Concerns: Long-term high-protein diets, particularly those high in animal proteins like red meat, have been linked to potential health risks, including kidney stones, kidney disease progression (in those with preexisting conditions), liver malfunction, colorectal cancer, and osteoporosis.
    Weight Loss and Regain: While high-protein diets may aid in weight loss, many individuals struggle to maintain them, leading to weight regain.
    Conclusion:
    A well-balanced high-protein diet can support muscle growth and maintenance, especially for older adults and athletes. However, whole food sources are preferable to supplements, and excessive consumption—especially from red meat—may carry health risks."""
]

# Construct dataset for evaluation
dataset = []
for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = rag.retrieve_relevant_context(index, query, embedding_model)
    response = rag.generate_answer(client, index, query, embedding_model)
    dataset.append({
        "user_input": query,
        "retrieved_contexts": relevant_docs,
        "response": response,
        "reference": reference
    })

evaluation_dataset = EvaluationDataset.from_list(dataset)

# Initialize LLM for evaluation
llm = ChatOpenAI(
    model_name="gpt-4o-mini",  # Choose "gpt-3.5-turbo", "gpt-4", or "gpt-4o"
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)
evaluator_llm = LangchainLLMWrapper(llm)

# Run evaluation
result = evaluate(dataset=evaluation_dataset, metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()], llm=evaluator_llm)

# Print evaluation results
print(result)