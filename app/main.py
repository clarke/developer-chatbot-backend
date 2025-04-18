from fastapi import FastAPI, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Codebase QA System")

# --- Configurable Settings ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "codebase")
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/llama-2-7b-chat.Q4_K_M.gguf"
)

# --- Load Model and Embedding ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32  # Process embeddings in batches
    }
)

# Initialize LlamaCpp LLM with optimized settings
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.2,  # Lower temperature for faster, more focused responses
    max_tokens=256,   # Reduce max tokens since we're doing focused QA
    n_ctx=1024,      # Reduce context window since we're doing focused QA
    n_threads=8,     # Use more threads for faster processing
    n_batch=512,     # Process tokens in larger batches
    verbose=False    # Disable verbose output for faster processing
)

# --- Connect to Qdrant ---
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
db = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model
)

# --- Setup QA Chain ---
prompt_template = (
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know. "
    "Don't try to make up an answer. Only use the information provided in "
    "the context. Do not use any external knowledge.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(
        search_kwargs={
            "k": 4,  # Reduce to 2 most relevant documents
            "score_threshold": 0.7  # Only return documents with high relevance
        }
    ),
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)


# --- Request Schema ---
class QueryRequest(BaseModel):
    question: str


# --- Endpoints ---
@app.post("/ask")
async def ask_codebase(query: QueryRequest):
    try:
        result = qa_chain(query.question)
        return {
            "answer": result["result"],
            "sources": [
                {
                    "source": doc.metadata.get("source"),
                    "snippet": doc.page_content[:200]
                }
                for doc in result["source_documents"]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}
