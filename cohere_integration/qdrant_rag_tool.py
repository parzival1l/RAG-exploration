"""Qdrant RAG Tool for Cohere Toolkit Integration.

This module provides a Cohere Toolkit compatible tool wrapper around the RAG pipeline.
It allows the Cohere Toolkit frontend to query the Qdrant-based RAG system.

Usage:
    1. Copy this file to: <cohere-toolkit>/src/community/tools/qdrant_rag_tool.py
    2. Update <cohere-toolkit>/src/community/tools/__init__.py to import QdrantRAGTool
    3. Register in <cohere-toolkit>/src/community/config/tools.py:
        class CommunityTool(Enum):
            Qdrant_RAG = QdrantRAGTool
    4. Add environment variables to Cohere Toolkit .env file
"""
import os
import sys
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports when running in Cohere Toolkit
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from backend.tools.base import BaseTool
    from backend.schemas.tool import ToolDefinition, ToolCategory
except ImportError:
    # Fallback for development/testing outside Cohere Toolkit
    BaseTool = object
    ToolDefinition = dict
    ToolCategory = type('ToolCategory', (), {'DataLoader': 'Data loader'})


class QdrantRAGTool(BaseTool):
    """RAG tool using Qdrant vector database and Replicate models.

    This tool retrieves relevant information from a Qdrant vector database
    to answer questions about technical documentation.
    """

    ID = "qdrant_rag_retriever"
    _pipeline = None  # Class-level cache for pipeline instance

    def __init__(self):
        """Initialize the RAG pipeline (lazy loading)."""
        pass

    @classmethod
    def _get_pipeline(cls):
        """Get or create the RAG pipeline instance (singleton pattern)."""
        if cls._pipeline is None:
            # Import here to avoid circular dependencies
            sys.path.insert(0, os.getenv('RAG_PIPELINE_PATH', '/home/user/RAG-exploration'))
            from src.rag_pipeline import create_rag_pipeline
            cls._pipeline = create_rag_pipeline()
        return cls._pipeline

    @classmethod
    def is_available(cls) -> bool:
        """Check if required environment variables are set.

        Returns:
            bool: True if all required credentials are present
        """
        required_vars = [
            "REPLICATE_API_TOKEN",
            "QDRANT_URL"
        ]

        available = all(os.getenv(var) for var in required_vars)

        if not available:
            missing = [var for var in required_vars if not os.getenv(var)]
            print(f"QdrantRAGTool unavailable. Missing: {', '.join(missing)}")

        return available

    @classmethod
    def get_tool_definition(cls) -> ToolDefinition:
        """Define the tool's interface and metadata.

        Returns:
            ToolDefinition: Tool configuration for Cohere Toolkit
        """
        return ToolDefinition(
            name=cls.ID,
            display_name="Qdrant RAG Retriever",
            description=(
                "Retrieves relevant information from a Qdrant vector database "
                "to answer questions about technical documentation. "
                "This tool searches through indexed documents and returns "
                "the most relevant passages with source citations. "
                "Best used for specific questions about documented topics."
            ),
            parameter_definitions={
                "query": {
                    "description": "The search query or question to find relevant information for",
                    "type": "str",
                    "required": True,
                },
                "num_results": {
                    "description": "Number of documents to retrieve (default: 5, max: 20)",
                    "type": "int",
                    "required": False,
                },
                "score_threshold": {
                    "description": "Minimum similarity score threshold between 0 and 1 (default: 0.7). Higher values return only more relevant results.",
                    "type": "float",
                    "required": False,
                }
            },
            is_visible=True,
            is_available=cls.is_available(),
            category=ToolCategory.DataLoader,
            error_message=cls.generate_error_message() if hasattr(cls, 'generate_error_message') else None,
        )

    async def call(
        self,
        parameters: dict,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute the RAG retrieval.

        Args:
            parameters: Dictionary containing:
                - query (str, required): The search query
                - num_results (int, optional): Number of results to return
                - score_threshold (float, optional): Minimum relevance score
            **kwargs: Additional context from Cohere Toolkit:
                - ctx: Request context
                - session: Database session
                - user_id: User identifier

        Returns:
            List[Dict[str, Any]]: List of document dictionaries with:
                - text (str, required): Document content
                - title (str, optional): Document title
                - url (str, optional): Source URL
        """
        query = parameters.get("query", "")
        num_results = parameters.get("num_results", 5)
        score_threshold = parameters.get("score_threshold")

        # Validate query
        if not query or not query.strip():
            if hasattr(self, 'get_no_results_error'):
                return self.get_no_results_error()
            return [{
                "text": "Error: No query provided",
                "title": "Input Error"
            }]

        # Validate num_results
        if num_results:
            num_results = min(max(int(num_results), 1), 20)  # Clamp between 1 and 20

        # Validate score_threshold
        if score_threshold:
            score_threshold = max(0.0, min(float(score_threshold), 1.0))  # Clamp between 0 and 1

        try:
            # Get pipeline instance
            pipeline = self._get_pipeline()

            # Retrieve documents using the RAG pipeline
            docs = pipeline.retrieve_documents(
                query=query,
                k=num_results,
                score_threshold=score_threshold
            )

            if not docs:
                if hasattr(self, 'get_no_results_error'):
                    return self.get_no_results_error()
                return [{
                    "text": f"No relevant documents found for query: {query}",
                    "title": "No Results"
                }]

            # Format results according to Cohere Toolkit contract
            results = []
            for idx, doc in enumerate(docs):
                metadata = doc.metadata

                result = {
                    "text": doc.page_content,
                }

                # Add optional fields if available in metadata
                # Try various common metadata field names
                if "title" in metadata:
                    result["title"] = str(metadata["title"])
                elif "name" in metadata:
                    result["title"] = str(metadata["name"])
                else:
                    result["title"] = f"Document {idx + 1}"

                if "source" in metadata:
                    result["url"] = str(metadata["source"])
                elif "url" in metadata:
                    result["url"] = str(metadata["url"])
                elif "link" in metadata:
                    result["url"] = str(metadata["link"])

                results.append(result)

            return results

        except Exception as e:
            error_msg = f"Failed to retrieve documents: {str(e)}"
            print(f"QdrantRAGTool error: {error_msg}")

            if hasattr(self, 'get_tool_error'):
                return self.get_tool_error(
                    error_type="RetrievalError",
                    message=error_msg
                )
            return [{
                "text": error_msg,
                "title": "Retrieval Error"
            }]


# For testing outside Cohere Toolkit
if __name__ == "__main__":
    import asyncio

    async def test_tool():
        """Test the QdrantRAGTool independently."""
        print("Testing Qdrant RAG Tool...")
        print(f"Available: {QdrantRAGTool.is_available()}")

        if QdrantRAGTool.is_available():
            tool = QdrantRAGTool()

            test_query = {
                "query": "What is Qdrant?",
                "num_results": 3
            }

            print(f"\nTest Query: {test_query['query']}")
            results = await tool.call(test_query)

            print(f"\nResults ({len(results)}):")
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Title: {result.get('title', 'N/A')}")
                print(f"URL: {result.get('url', 'N/A')}")
                print(f"Text: {result['text'][:200]}...")
        else:
            print("\nTool not available. Check environment variables:")
            print("  - REPLICATE_API_TOKEN")
            print("  - QDRANT_URL")
            print("  - QDRANT_API_KEY (optional)")
            print("  - COLLECTION_NAME (optional)")

    asyncio.run(test_tool())
