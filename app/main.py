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
# --- Revised Prompt Template ---
prompt_template = (
    "You are a code assistant answering questions strictly based on the following code snippets.\n\n"  # noqa: E501
    "Only answer using the provided snippets.\n"
    "If the answer is not in the code snippets, reply with:\n"
    "'I don't know based on the provided code.'\n\n"
    "Code Snippets:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# --- Updated Retriever Setup ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(
        search_kwargs={
            "k": 4  # Removed score_threshold for better recall
        }
    ),
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": PROMPT,
        "document_separator": "\n\n---\n\n"
    }
)


# --- Request Schema ---
class QueryRequest(BaseModel):
    question: str


# --- Endpoints ---
# --- Updated Endpoint with Debug Logging ---
@app.post("/ask")
async def ask_codebase(query: QueryRequest):
    try:
        result = qa_chain(query.question)

        # 🔍 Log retrieved documents for debug
        print("\n\n=== Retrieved Documents ===")
        for doc in result["source_documents"]:
            print(doc.metadata.get("source", "No source"), "—",
                  doc.page_content[:300])
            print("---")

        sources = []
        for doc in result["source_documents"]:
            source = doc.metadata.get("source")
            if source:
                sources.append({
                    "source": source,
                    "snippet": doc.page_content[:200]
                })

        return {
            "answer": result["result"],
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}
