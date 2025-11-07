"""Custom Replicate embeddings wrapper for LangChain."""
import json
from typing import List
import replicate
from langchain_core.embeddings import Embeddings
from config.settings import EMBEDDING_MODEL, BATCH_SIZE, REPLICATE_API_TOKEN


class ReplicateEmbeddings(Embeddings):
    """Custom embeddings class using Replicate's multilingual-e5-large model."""

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = BATCH_SIZE,
        api_token: str = REPLICATE_API_TOKEN,
    ):
        """Initialize the Replicate embeddings.

        Args:
            model: The Replicate model identifier
            batch_size: Batch size for processing multiple texts
            api_token: Replicate API token
        """
        self.model = model
        self.batch_size = batch_size
        self.api_token = api_token

        # Set the API token
        if api_token:
            import os
            os.environ["REPLICATE_API_TOKEN"] = api_token

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Format texts as JSON string (as required by the model)
            texts_json = json.dumps(batch)

            try:
                output = replicate.run(
                    self.model,
                    input={
                        "texts": texts_json,
                        "batch_size": self.batch_size,
                        "normalize_embeddings": True
                    }
                )

                # The output is already a list of embeddings
                if isinstance(output, list):
                    all_embeddings.extend(output)
                else:
                    raise ValueError(f"Unexpected output format from Replicate: {type(output)}")

            except Exception as e:
                raise RuntimeError(f"Error getting embeddings from Replicate: {str(e)}")

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of documents to embed

        Returns:
            List of embeddings, one for each document
        """
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector for the query
        """
        embeddings = self._get_embeddings([text])
        return embeddings[0]


def get_embeddings() -> ReplicateEmbeddings:
    """Factory function to create embeddings instance.

    Returns:
        Configured ReplicateEmbeddings instance
    """
    return ReplicateEmbeddings()
