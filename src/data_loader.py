"""Data loading and indexing module for the RAG pipeline."""
import uuid
from typing import List, Dict, Any
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

from config.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATASET_NAME,
    DATASET_SPLIT,
)
from src.embeddings import get_embeddings


class DataLoader:
    """Handles loading, processing, and indexing documents into Qdrant."""

    def __init__(self):
        """Initialize the data loader with Qdrant client and embeddings."""
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.embeddings = get_embeddings()
        self.collection_name = COLLECTION_NAME

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_dataset_from_hf(self) -> List[Dict[str, Any]]:
        """Load dataset from Hugging Face.

        Returns:
            List of documents from the dataset
        """
        print(f"Loading dataset: {DATASET_NAME}")
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        print(f"Loaded {len(dataset)} documents")

        return [dict(item) for item in dataset]

    def create_chunks(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """Split documents into chunks.

        Args:
            documents: List of raw documents

        Returns:
            List of LangChain Document objects (chunks)
        """
        print("Creating text chunks...")
        all_chunks = []

        for doc in tqdm(documents, desc="Processing documents"):
            # Extract text content - adjust field names based on your dataset structure
            # Common field names: 'text', 'content', 'page_content', 'document'
            text_content = self._extract_text(doc)

            if not text_content:
                continue

            # Create metadata
            metadata = {k: v for k, v in doc.items() if k != self._get_text_field(doc)}
            metadata["source_id"] = str(uuid.uuid4())

            # Split into chunks
            chunks = self.text_splitter.create_documents(
                texts=[text_content],
                metadatas=[metadata]
            )

            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def _get_text_field(self, doc: Dict[str, Any]) -> str:
        """Identify the text field in the document.

        Args:
            doc: Document dictionary

        Returns:
            Name of the text field
        """
        possible_fields = ['text', 'content', 'page_content', 'document', 'body']
        for field in possible_fields:
            if field in doc:
                return field
        # Return first string field as fallback
        for key, value in doc.items():
            if isinstance(value, str) and len(value) > 50:
                return key
        return list(doc.keys())[0]

    def _extract_text(self, doc: Dict[str, Any]) -> str:
        """Extract text content from a document.

        Args:
            doc: Document dictionary

        Returns:
            Text content
        """
        text_field = self._get_text_field(doc)
        return str(doc.get(text_field, ""))

    def create_collection(self, recreate: bool = False):
        """Create Qdrant collection for storing embeddings.

        Args:
            recreate: Whether to recreate the collection if it exists
        """
        if recreate and self.client.collection_exists(self.collection_name):
            print(f"Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            print("Collection created successfully")
        else:
            print(f"Collection {self.collection_name} already exists")

    def index_documents(self, chunks: List[Document], batch_size: int = 10):
        """Index document chunks into Qdrant.

        Args:
            chunks: List of document chunks to index
            batch_size: Number of chunks to process at once
        """
        print(f"Indexing {len(chunks)} chunks into Qdrant...")

        points = []

        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding and indexing"):
            batch = chunks[i:i + batch_size]

            # Get texts and metadatas
            texts = [chunk.page_content for chunk in batch]

            # Get embeddings for batch
            try:
                embeddings = self.embeddings.embed_documents(texts)
            except Exception as e:
                print(f"Error embedding batch {i}: {str(e)}")
                continue

            # Create points for Qdrant
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                point_id = str(uuid.uuid4())
                payload = {
                    "text": chunk.page_content,
                    "metadata": chunk.metadata,
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            # Upload batch to Qdrant
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                points = []

        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        print("Indexing completed successfully")

    def run_full_pipeline(self, recreate_collection: bool = False):
        """Run the complete data loading and indexing pipeline.

        Args:
            recreate_collection: Whether to recreate the collection
        """
        print("=" * 50)
        print("Starting RAG Data Indexing Pipeline")
        print("=" * 50)

        # Load dataset
        documents = self.load_dataset_from_hf()

        # Create chunks
        chunks = self.create_chunks(documents)

        # Create collection
        self.create_collection(recreate=recreate_collection)

        # Index documents
        self.index_documents(chunks)

        print("=" * 50)
        print("Pipeline completed successfully!")
        print(f"Collection: {self.collection_name}")
        print(f"Total chunks indexed: {len(chunks)}")
        print("=" * 50)


if __name__ == "__main__":
    loader = DataLoader()
    loader.run_full_pipeline(recreate_collection=True)
