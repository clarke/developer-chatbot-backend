#!/usr/bin/env python3

import os
import glob
import subprocess
import argparse
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
import torch

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

load_dotenv()

# --- Config ---
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "codebase")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
FILE_BATCH_SIZE = 64  # Increased from 32 to 64
EMBEDDING_BATCH_SIZE = 64
# For M1 Max: 10 cores * 2 threads per core = 20 workers
MAX_WORKERS = min(20, (os.cpu_count() or 4) * 2)  # I/O bound tasks

# Directories to ignore when collecting files
IGNORED_DIRS = {
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
}

# File extension -> language
EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".php": "php",
    ".ts": "ts",
    ".tsx": "ts",
    ".js": "js",
    ".jsx": "js",
    ".html": "html",
    ".md": "markdown",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".cs": "csharp",
    ".sol": "sol",
    ".lua": "lua",
    ".pl": "perl",
    ".hs": "haskell",
    ".ex": "elixir",
    ".ps1": "powershell"
}


def should_ignore_path(path: str) -> bool:
    """Check if a path should be ignored based on the ignored
    directories list."""
    path_parts = path.split(os.sep)
    return any(ignored_dir in path_parts for ignored_dir in IGNORED_DIRS)


def process_file(file_path: str) -> Dict[str, Any]:
    """Process a single file and return its document."""
    try:
        loader = TextLoader(file_path)
        doc = loader.load()[0]
        ext = os.path.splitext(file_path)[1]
        doc.metadata.update({
            "language": EXTENSION_LANGUAGE_MAP.get(ext, "text"),
            "filename": os.path.basename(file_path),
            "source": file_path,
            "repo": os.path.basename(
                os.path.dirname(os.path.dirname(file_path))
            ),
            "commit_sha": "local"  # For local files
        })
        return {"success": True, "doc": doc}
    except Exception as e:
        return {"success": False, "error": str(e), "file": file_path}


def process_files_batch(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Process a batch of files in parallel."""
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(process_file, file_paths))


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


def generate_embeddings_batch(
    texts: List[str],
    embeddings: HuggingFaceEmbeddings,
    batch_size: int = EMBEDDING_BATCH_SIZE
) -> List[List[float]]:
    """Generate embeddings for a batch of texts in parallel."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Ingest source code into Qdrant."
    )
    parser.add_argument(
        "repo_dir",
        help="Path to the repository to ingest"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh Qdrant collection before ingesting"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.repo_dir):
        raise ValueError(
            f"Repository directory does not exist: {args.repo_dir}"
        )

    # --- File Collection ---
    print("Collecting files...")
    all_files = []
    for ext in EXTENSION_LANGUAGE_MAP.keys():
        pattern = os.path.join(args.repo_dir, f"**/*{ext}")
        files = glob.glob(pattern, recursive=True)
        # Filter out files in ignored directories
        files = [f for f in files if not should_ignore_path(f)]
        all_files.extend(files)

    # --- Parallel File Processing ---
    print(
        f"Processing {len(all_files)} files in parallel "
        f"using {MAX_WORKERS} workers..."
    )
    documents = []
    for i in tqdm(range(0, len(all_files), FILE_BATCH_SIZE)):
        batch = all_files[i:i + FILE_BATCH_SIZE]
        results = process_files_batch(batch)
        for result in results:
            if result["success"]:
                documents.append(result["doc"])
            else:
                print(f"Failed to load {result['file']}: {result['error']}")

    # --- Chunking ---
    print("Chunking documents...")
    chunks = []
    for doc in tqdm(documents):
        language = doc.metadata.get("language", "text")
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = splitter.split_documents([doc])
        for chunk in split_docs:
            # Preserve all metadata from the original document
            chunk.metadata.update(doc.metadata)
        chunks.extend(split_docs)

    # --- Embedding + Qdrant ---
    print("Initializing embeddings and Qdrant...")

    # Configure device for Apple Silicon
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("Using CPU (MPS not available)")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if args.refresh:
        print(f"Recreating collection '{COLLECTION_NAME}'...")
        try:
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        except Exception as e:
            print(f"Warning: Failed to delete existing collection: {e}")

        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE
            )
        )

    # --- Parallel Embedding Generation and Storage ---
    print("Generating embeddings and storing in Qdrant...")
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Generate embeddings in parallel
    vectors = generate_embeddings_batch(texts, embeddings)

    # Upload to Qdrant in batches
    points = []
    for i in tqdm(range(0, len(chunks), EMBEDDING_BATCH_SIZE)):
        batch_vectors = vectors[i:i + EMBEDDING_BATCH_SIZE]
        batch_metadatas = metadatas[i:i + EMBEDDING_BATCH_SIZE]
        batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]

        batch_points = [
            models.PointStruct(
                id=i + j,
                vector=vector,
                payload={
                    "text": text,
                    "metadata": metadata
                }
            )
            for j, (vector, text, metadata) in enumerate(
                zip(batch_vectors, batch_texts, batch_metadatas)
            )
        ]
        points.extend(batch_points)

        # Upload in larger batches to reduce API calls
        if len(points) >= EMBEDDING_BATCH_SIZE * 4:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            points = []

    # Upload any remaining points
    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    print(
        f"Ingested {len(chunks)} chunks from {len(documents)} files."
    )


if __name__ == "__main__":
    main()
