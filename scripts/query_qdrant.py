#!/usr/bin/env python3

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to your Qdrant instance
client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333))
)

# Embed your query
embedder = HuggingFaceEmbeddings()
query = "How is logging configured in this codebase?"
query_vector = embedder.embed_query(query)

collection_name = os.getenv("QDRANT_COLLECTION", "codebase")

search_result = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=5,
    with_payload=True,
    with_vectors=False
)


for hit in search_result:
    print("Score:", hit.score)
    print("Payload:", hit.payload)
    print("---")
