#!/usr/bin/env python3
import argparse
import sys
from typing import Optional
import requests
from pydantic import BaseModel


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, str]]


def query_api(
    question: str,
    base_url: str = "http://localhost:8000"
) -> Optional[QueryResponse]:
    """
    Query the codebase QA API with a question.

    Args:
        question: The question to ask
        base_url: Base URL of the API server

    Returns:
        QueryResponse if successful, None if there was an error
    """
    try:
        response = requests.post(
            f"{base_url}/ask",
            json={"question": question},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return QueryResponse(**response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error querying API: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Query the codebase QA API")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API server"
    )
    args = parser.parse_args()

    result = query_api(args.question, args.url)
    if result:
        print("\nAnswer:")
        print(result.answer)

        if result.sources:
            print("\nSources:")
            for i, source in enumerate(result.sources, 1):
                print(f"\n{i}. {source['source']}")
                print(f"   {source['snippet']}...")


if __name__ == "__main__":
    main()
