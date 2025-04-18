#!/usr/bin/env python3

"""
This script is used to inspect the contents of the Qdrant collection.
It can be used to get a summary of the collection, or to show a sample
of the collection.

Usage:
  ./scripts/inspect_qdrant.py
  ./scripts/inspect_qdrant.py --repo <repo_name>
  ./scripts/inspect_qdrant.py --language <language>
  ./scripts/inspect_qdrant.py --repo <repo_name> --language <language>

Example:
  ./scripts/inspect_qdrant.py --repo repo-dir --language python
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import os


load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "codebase")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_metadata_summary():
    print(f"\n📦 Summary for collection: '{COLLECTION_NAME}'")

    points = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,  # adjust for more
        with_payload=True
    )[0]

    repos = set()
    languages = set()
    for pt in points:
        payload = pt.payload or {}
        repos.add(payload.get("repo", "unknown"))
        languages.add(payload.get("language", "unknown"))

    print(f"🔹 Repos Found: {repos}")
    print(f"🔹 Languages Found: {languages}")
    print(f"🔹 Total Chunks Inspected: {len(points)}")


def show_samples(repo: str = None, language: str = None, limit: int = 5):
    conditions = []

    if repo:
        conditions.append(
            FieldCondition(key="repo", match=MatchValue(value=repo))
        )
    if language:
        conditions.append(
            FieldCondition(key="language", match=MatchValue(value=language))
        )

    filter_obj = Filter(must=conditions) if conditions else None

    print(f"\n📄 Showing up to {limit} samples...")
    
    # Use search instead of scroll for filtered queries
    if filter_obj:
        points = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=[0] * 768,  # Dummy vector for filtering
            limit=limit,
            with_payload=True,
            query_filter=filter_obj
        )
    else:
        points = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            with_payload=True
        )[0]

    for idx, pt in enumerate(points, 1):
        payload = pt.payload or {}
        print(f"\n--- Document {idx} ---")
        print(
            f"File: {payload.get('filename')} | "
            f"Lang: {payload.get('language')} | "
            f"Repo: {payload.get('repo')}"
        )
        print(
            f"SHA: {payload.get('commit_sha')} | "
            f"Source: {payload.get('source')}"
        )
        snippet = payload.get("text") or payload.get("page_content") or "<No content>"  # noqa: E501
        print(f"Snippet: {snippet[:300].strip()}...")
    print("\n--- End of Samples ---\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect Qdrant collection contents."
    )
    parser.add_argument(
        "--repo", help="Filter by repository name", default=None
    )
    parser.add_argument(
        "--language", help="Filter by language (e.g., python, php)",
        default=None
    )
    parser.add_argument(
        "--limit", help="Number of samples to show", type=int, default=5
    )

    args = parser.parse_args()

    get_metadata_summary()
    show_samples(repo=args.repo, language=args.language, limit=args.limit)
