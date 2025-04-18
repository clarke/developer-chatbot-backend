from fastapi import FastAPI, HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Add this to your .env file

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# --- Load Model and Embedding ---
embedding_model = OpenAIEmbeddings()

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=512
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
    retriever=db.as_retriever(),
    return_source_documents=True
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
