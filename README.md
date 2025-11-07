# RAG Pipeline with Qdrant and Replicate

A production-ready Retrieval-Augmented Generation (RAG) pipeline that combines vector search, document retrieval, and language model generation. Designed to integrate seamlessly with the **Cohere Toolkit** frontend for a complete RAG application.

## Overview

This RAG system uses:
- **Qdrant Cloud** - Remote vector database cluster for efficient similarity search
- **Replicate** - API access to embedding models (multilingual-e5-large) and LLMs (Llama 3)
- **LangChain** - Orchestration and pipeline management
- **Cohere Toolkit** - Modern chat-based frontend interface

> **Note**: This system connects to your existing Qdrant cluster. No local database setup required!

## Architecture

### Full Stack with Cohere Toolkit

```
User Interface (Cohere Toolkit Frontend)
    ↓
FastAPI Backend (Cohere Toolkit)
    ↓
QdrantRAGTool (this integration)
    ↓
RAG Pipeline (this repository)
    ↓
┌─────────────────────────────────────┐
│ Query Embedding (Replicate E5)      │
│          ↓                           │
│ Vector Search (Your Qdrant Cluster) │
│          ↓                           │
│ Context Retrieval & Ranking          │
│          ↓                           │
│ LLM Generation (Replicate Llama 3)  │
└─────────────────────────────────────┘
    ↓
Response with Source Citations
```

### Key Components

1. **Embeddings** (`src/embeddings.py`)
   - Custom LangChain wrapper for Replicate's multilingual-e5-large
   - Batch processing for efficient embedding generation
   - Normalized embeddings for cosine similarity search

2. **Data Loader** (`src/data_loader.py`)
   - Loads datasets from Hugging Face
   - Chunks documents using RecursiveCharacterTextSplitter
   - Indexes into your Qdrant cluster with metadata

3. **RAG Pipeline** (`src/rag_pipeline.py`)
   - LangChain LCEL-based pipeline
   - Connects to your remote Qdrant cluster
   - Configurable retrieval parameters
   - Source attribution in responses

4. **Query Interface** (`src/query_interface.py`)
   - Interactive CLI mode for testing
   - Single question and batch processing modes
   - Source document display

5. **Cohere Integration** (`cohere_integration/`)
   - Cohere Toolkit tool wrapper (`qdrant_rag_tool.py`)
   - Full integration guide for frontend setup
   - API-compatible response formatting

## Installation

### Prerequisites

- **Python 3.8+**
- **Existing Qdrant cluster** (Qdrant Cloud or self-hosted)
  - Cluster URL (e.g., `https://your-cluster.qdrant.io:6333`)
  - API key for authentication
  - Collection name where documents are/will be stored
- **Replicate API account**
  - Sign up at https://replicate.com
  - Get API token from https://replicate.com/account/api-tokens
- **Cohere Toolkit** (for frontend - optional for standalone use)
  - Clone from: https://github.com/parzival1l/cohere-toolkit/

### Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/parzival1l/RAG-exploration.git
   cd RAG-exploration
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your credentials:
   ```env
   # Your Replicate API token
   REPLICATE_API_TOKEN=your_replicate_token_here

   # Your Qdrant cluster details
   QDRANT_URL=https://your-cluster.qdrant.io:6333
   QDRANT_API_KEY=your_qdrant_api_key_here
   COLLECTION_NAME=your_collection_name_here
   ```

4. **Verify configuration**
   ```bash
   python main.py stats
   ```

   This should connect to your Qdrant cluster and show collection statistics.

## Usage

### Standalone Mode (Testing & Development)

#### 1. Index Documents

If your collection is empty, index the sample Qdrant documentation dataset:

```bash
python main.py index
```

To recreate the collection (⚠️ deletes existing data):
```bash
python main.py index --recreate
```

**What happens:**
- Downloads dataset from Hugging Face (`atitaarora/qdrant_doc`)
- Splits documents into 512-token chunks with 50-token overlap
- Generates embeddings using Replicate's multilingual-e5-large
- Stores in your Qdrant cluster with metadata

#### 2. Query the System

**Interactive Mode** (recommended for testing):
```bash
python main.py query
```

Commands in interactive mode:
- Type your question and press Enter
- `sources on` - Show source documents
- `sources off` - Hide source documents
- `stats` - Show collection statistics
- `exit` or `quit` - Exit

**Single Question Mode**:
```bash
python main.py query -q "What is Qdrant?"
```

**Batch Mode** (process multiple questions):
```bash
# Create a file with questions (one per line)
echo "What is Qdrant?" > questions.txt
echo "How does vector search work?" >> questions.txt

# Process batch
python main.py query -b questions.txt -o results.json
```

#### 3. Check Collection Statistics

```bash
python main.py stats
```

Shows your Qdrant collection info, document count, and status.

### Production Mode (with Cohere Toolkit Frontend)

For the full chat interface with UI, integrate with Cohere Toolkit:

1. **Follow the integration guide**:
   ```bash
   cat cohere_integration/README.md
   ```

2. **Quick setup**:
   ```bash
   # Copy the tool to Cohere Toolkit
   cp cohere_integration/qdrant_rag_tool.py <cohere-toolkit>/src/community/tools/

   # Register and configure (see integration guide for details)
   # Then start Cohere Toolkit
   cd <cohere-toolkit>
   make dev
   ```

3. **Access the UI**:
   - Open http://localhost:4000
   - Enable "Qdrant RAG Retriever" in tools
   - Start chatting with your documents!

📖 **Full integration guide**: See `cohere_integration/README.md` for detailed setup instructions, troubleshooting, and advanced configuration.

## Configuration

All configuration is in `.env`. Key parameters:

### Embedding Configuration
```env
EMBEDDING_MODEL=beautyyuyanli/multilingual-e5-large:...
EMBEDDING_DIMENSION=1024
BATCH_SIZE=32
```

### Text Splitting
```env
CHUNK_SIZE=512          # Tokens per chunk
CHUNK_OVERLAP=50        # Overlap between chunks
```

### Retrieval
```env
TOP_K_CANDIDATES=15     # Number of documents to retrieve
SCORE_THRESHOLD=0.7     # Minimum similarity score
```

### Generation
```env
LLM_MODEL=meta/meta-llama-3-70b-instruct
TEMPERATURE=0.2         # Lower = more factual
MAX_TOKENS=1000         # Response length limit
```

## Advanced Usage

### Custom Dataset

To use your own dataset, modify `src/data_loader.py`:

```python
# In DataLoader.load_dataset_from_hf()
dataset = load_dataset("your-dataset-name", split="train")
```

Or load from local files:
```python
def load_custom_documents(self) -> List[Dict[str, Any]]:
    documents = []
    for file in Path("data/").glob("*.txt"):
        with open(file) as f:
            documents.append({"text": f.read(), "source": file.name})
    return documents
```

### Programmatic Usage

```python
from src.rag_pipeline import create_rag_pipeline

# Initialize pipeline
pipeline = create_rag_pipeline()

# Query
result = pipeline.query("What is Qdrant?", return_sources=True)

print(result['answer'])
for source in result['sources']:
    print(f"Source: {source['metadata']}")
```

### Custom Prompts

Edit the prompt template in `src/rag_pipeline.py`:

```python
def _create_prompt(self):
    template = """Your custom prompt here...

    Context: {context}
    Question: {question}
    Answer:"""
    return ChatPromptTemplate.from_template(template)
```

## Project Structure

```
RAG-exploration/
├── config/
│   └── settings.py              # Configuration management
├── src/
│   ├── embeddings.py            # Replicate embeddings wrapper
│   ├── data_loader.py           # Data loading and indexing
│   ├── rag_pipeline.py          # RAG pipeline implementation
│   ├── query_interface.py       # Interactive query interface
│   └── evaluation.py            # RAGAS evaluation (banked for later)
├── cohere_integration/
│   ├── qdrant_rag_tool.py       # Cohere Toolkit integration
│   └── README.md                # Integration guide
├── main.py                      # Main CLI entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## Troubleshooting

### "No module named 'config'"

Ensure you're running from the project root:
```bash
cd RAG-exploration
python main.py query
```

### "Missing required environment variables"

Copy and configure the environment file:
```bash
cp .env.example .env
# Edit .env with your credentials (REPLICATE_API_TOKEN, QDRANT_URL, etc.)
```

### "Error getting embeddings from Replicate"

Check your Replicate API token:
```bash
# Test the token
python -c "import replicate; print('Token is valid')"
```

Or visit: https://replicate.com/account/api-tokens

### Cannot connect to Qdrant cluster

Check your Qdrant configuration:
```bash
# Test connection
curl "https://your-cluster.qdrant.io:6333/collections" \
  -H "api-key: your_api_key"
```

Verify:
- URL includes protocol (`https://`) and port (`:6333`)
- API key is correct
- Firewall allows connection
- For Qdrant Cloud, cluster is active

### "Collection does not exist"

Either:
1. Index documents: `python main.py index`
2. Or update `COLLECTION_NAME` in `.env` to match your existing collection

### Cohere Toolkit integration issues

See `cohere_integration/README.md` for detailed troubleshooting of frontend integration.

## Performance Optimization

### Batch Size Tuning

Larger batches = faster embedding but more memory:
```env
BATCH_SIZE=64  # Increase if you have sufficient memory
```

### Retrieval Parameters

Adjust based on your use case:
```env
TOP_K_CANDIDATES=20     # More candidates = better recall, slower
SCORE_THRESHOLD=0.6     # Lower threshold = more results
```

### Chunk Size

Optimal chunk size depends on your content:
- **Technical docs**: 512-768 tokens
- **Conversational**: 256-512 tokens
- **Long-form**: 768-1024 tokens

## Evaluation

RAGAS-based evaluation is available but currently banked for later use. To enable:

1. Uncomment evaluation sections in `main.py`
2. Install OpenAI dependencies (required by RAGAS)
3. Set `OPENAI_API_KEY` environment variable
4. Run: `python main.py evaluate --generate-test-set`

For now, you can evaluate the system manually by:
- Testing with known questions
- Reviewing source attributions
- Checking answer accuracy
- Using the Cohere Toolkit UI for interactive testing

## Contributing

Contributions welcome! Areas for improvement:
- Support for additional embedding models
- Query expansion and reformulation
- Re-ranking strategies
- Multi-modal support (images, tables)
- Streaming responses

## License

MIT License - see LICENSE.txt file for details

## Acknowledgments

- **Qdrant** - High-performance vector database
- **Replicate** - Easy API access to ML models
- **LangChain** - Powerful RAG orchestration
- **Cohere Toolkit** - Modern chat interface framework
- **Dataset** - `atitaarora/qdrant_doc` from Hugging Face

## Support

For issues and questions:

**RAG Pipeline Issues:**
- Check troubleshooting section above
- Review configuration in `.env`
- Test connection: `python main.py stats`
- Check Qdrant and Replicate documentation

**Cohere Integration Issues:**
- See `cohere_integration/README.md`
- Check Cohere Toolkit logs: `docker-compose logs backend`
- Verify tool registration: `curl http://localhost:8000/v1/tools`

**General:**
- Open an issue on GitHub: https://github.com/parzival1l/RAG-exploration/issues

## Roadmap

- [x] Qdrant Cloud/remote cluster support
- [x] Cohere Toolkit frontend integration
- [x] Replicate embeddings and LLM integration
- [ ] Add support for multi-query retrieval
- [ ] Implement hybrid search (vector + keyword)
- [ ] Add conversation memory
- [ ] Support for multiple collections
- [ ] RAGAS evaluation re-enablement
- [ ] Caching layer for frequent queries
- [ ] Production deployment guide
