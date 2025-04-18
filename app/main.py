from fastapi import FastAPI, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
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
    encode_kwargs={"normalize_embeddings": True}
)

# Initialize LlamaCpp LLM
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.2,
    max_tokens=512,
    n_ctx=2048,
    n_threads=4,
    verbose=True
)

# --- Connect to Qdrant ---
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
db = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model
)

# --- Setup QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(
        search_kwargs={"k": 4}  # Return top 4 most relevant documents
    ),
    return_source_documents=True,
    chain_type="stuff"
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
