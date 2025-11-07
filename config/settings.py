"""Configuration settings for the RAG pipeline."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Replicate Configuration
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "qdrant_docs")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "beautyyuyanli/multilingual-e5-large:a06276a89f1a902d5fc225a9ca32b6e8e6292b7f3b136518878da97c458e2bad"
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Text Splitting Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval Configuration
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "15"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.7"))

# Generation Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "meta/meta-llama-3-70b-instruct")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# Dataset Configuration
DATASET_NAME = os.getenv("DATASET_NAME", "atitaarora/qdrant_doc")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")


def validate_config():
    """Validate that all required configuration is present."""
    required_vars = {
        "REPLICATE_API_TOKEN": REPLICATE_API_TOKEN,
    }

    missing = [key for key, value in required_vars.items() if not value]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please copy .env.example to .env and fill in your credentials."
        )

    return True
