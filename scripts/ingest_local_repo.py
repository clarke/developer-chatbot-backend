#!/usr/bin/env python3

import os
import glob
import subprocess
import argparse
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

# --- Config ---
REPO_DIR = "../fleetsu-app"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "codebase")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# File extension -> language
EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".php": "php",
    ".ts": "ts",
    ".js": "js",
    # ".sql": "sql",
    ".html": "html",
    # ".json": "json",
    # ".twig": "twig"
}


# --- Git Metadata ---
def get_repo_metadata(repo_path: str):
    try:
        repo_name = os.path.basename(os.path.abspath(repo_path))
        commit_sha = subprocess.check_output(
            ["git", "-C", repo_path, "rev-parse", "HEAD"]
        ).decode().strip()
        return repo_name, commit_sha
    except Exception as e:
        print(f"Warning: Failed to get Git metadata - {e}")
        return "unknown", "unknown"


# --- Args ---
parser = argparse.ArgumentParser(description="Ingest source code into Qdrant.")
parser.add_argument("--refresh", action="store_true",
                    help="Refresh Qdrant collection before ingesting")
args = parser.parse_args()

# --- File Collection ---
all_files = []
for ext in EXTENSION_LANGUAGE_MAP.keys():
    pattern = os.path.join(REPO_DIR, f"**/*{ext}")
    all_files.extend(glob.glob(pattern, recursive=True))

documents = []
for file_path in all_files:
    try:
        loader = TextLoader(file_path)
        doc = loader.load()[0]
        ext = os.path.splitext(file_path)[1]
        doc.metadata["language"] = EXTENSION_LANGUAGE_MAP.get(ext, "text")
        doc.metadata["filename"] = os.path.basename(file_path)
        doc.metadata["source"] = file_path
        documents.append(doc)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")

# --- Git Metadata ---
repo_name, commit_sha = get_repo_metadata(REPO_DIR)

# --- Chunking ---
chunks = []
for doc in documents:
    language = doc.metadata.get("language", "text")
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=1000,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents([doc])
    for chunk in split_docs:
        chunk.metadata["repo"] = repo_name
        chunk.metadata["commit_sha"] = commit_sha
    chunks.extend(split_docs)

# --- Embedding + Qdrant ---
embeddings = OpenAIEmbeddings()
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

if args.refresh:
    print(f"Refreshing collection '{COLLECTION_NAME}'...")
    try:
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception as e:
        print(f"Warning: Failed to delete existing collection: {e}")

try:
    qdrant_client.get_collection(collection_name=COLLECTION_NAME)
except Exception:
    print(f"Creating collection '{COLLECTION_NAME}'...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE
        )
    )

db = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)
db.add_documents(chunks)
print(
    f"Ingested {len(chunks)} chunks from {len(documents)} files into Qdrant."
)
