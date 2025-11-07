# RAG Pipeline with Qdrant and Replicate

A production-ready Retrieval-Augmented Generation (RAG) pipeline that combines vector search, document retrieval, and language model generation to answer questions based on a knowledge base.

## Overview

This RAG system uses:
- **Qdrant** - Vector database for efficient similarity search
- **Replicate** - API access to embedding models and LLMs
- **LangChain** - Orchestration and pipeline management
- **RAGAS** - Evaluation framework for RAG performance metrics

## Architecture

```
User Query
    ↓
Query Embedding (Replicate E5-Large)
    ↓
Vector Search (Qdrant)
    ↓
Context Retrieval & Ranking
    ↓
LLM Generation (Replicate Llama 3)
    ↓
Response with Source Attribution
```

### Key Components

1. **Embeddings** (`src/embeddings.py`)
   - Custom LangChain wrapper for Replicate's multilingual-e5-large model
   - Batch processing for efficient embedding generation
   - Normalized embeddings for cosine similarity

2. **Data Loader** (`src/data_loader.py`)
   - Loads dataset from Hugging Face
   - Chunks documents using RecursiveCharacterTextSplitter
   - Indexes into Qdrant with metadata preservation

3. **RAG Pipeline** (`src/rag_pipeline.py`)
   - LangChain LCEL-based pipeline
   - Configurable retrieval parameters
   - Source attribution in responses

4. **Query Interface** (`src/query_interface.py`)
   - Interactive CLI mode
   - Single question mode
   - Batch processing mode

5. **Evaluation** (`src/evaluation.py`)
   - RAGAS-based metrics (faithfulness, relevancy, precision, recall)
   - Test set generation and management
   - Detailed performance reporting

## Installation

### Prerequisites

- Python 3.8+
- Qdrant server (local or cloud)
- Replicate API account

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG-exploration
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Qdrant**

   **Option A: Docker (Recommended)**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

   **Option B: Qdrant Cloud**
   - Sign up at https://cloud.qdrant.io
   - Create a cluster and note the URL and API key

4. **Configure environment**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your credentials:
   ```env
   REPLICATE_API_TOKEN=your_replicate_token_here
   QDRANT_URL=http://localhost:6333  # or your Qdrant Cloud URL
   QDRANT_API_KEY=your_qdrant_key_here  # optional for local
   ```

5. **Get Replicate API Token**
   - Sign up at https://replicate.com
   - Go to https://replicate.com/account/api-tokens
   - Create a token and add it to `.env`

## Usage

### 1. Index Documents

Load and index the Qdrant documentation dataset:

```bash
python main.py index
```

To recreate the collection (deletes existing data):
```bash
python main.py index --recreate
```

**What happens:**
- Downloads dataset from Hugging Face (`atitaarora/qdrant_doc`)
- Splits documents into 512-token chunks with 50-token overlap
- Generates embeddings using Replicate's multilingual-e5-large
- Stores in Qdrant with metadata

### 2. Query the System

**Interactive Mode** (recommended for exploration):
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

### 3. Evaluate Performance

**Generate a test set**:
```bash
python main.py evaluate --generate-test-set --num-questions 10
```

This creates `test_set.json`. **Review and customize** the questions and ground truth answers.

**Run evaluation**:
```bash
python main.py evaluate --test-set test_set.json
```

**RAGAS Metrics**:
- **Faithfulness** (>0.80): Responses grounded in retrieved context
- **Answer Relevancy** (>0.85): Directly addresses the question
- **Context Precision** (>0.70): Retrieved chunks are relevant
- **Context Recall** (>0.70): Important information not missed

### 4. Check Statistics

```bash
python main.py stats
```

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
│   └── settings.py          # Configuration management
├── src/
│   ├── embeddings.py        # Replicate embeddings wrapper
│   ├── data_loader.py       # Data loading and indexing
│   ├── rag_pipeline.py      # RAG pipeline implementation
│   ├── query_interface.py   # Interactive query interface
│   └── evaluation.py        # RAGAS evaluation
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
└── README.md               # This file
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
# Edit .env with your credentials
```

### "Error getting embeddings from Replicate"

Check your Replicate API token:
```bash
# Test the token
python -c "import replicate; print(replicate.Client().list_models())"
```

### "Collection does not exist"

Index documents first:
```bash
python main.py index
```

### RAGAS evaluation fails

RAGAS requires additional configuration for OpenAI API (used internally for evaluation). You can:
1. Set `OPENAI_API_KEY` environment variable
2. Or skip evaluation and use the pipeline directly

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

## Evaluation Metrics

### Target Scores

| Metric | Target | Description |
|--------|--------|-------------|
| Faithfulness | >0.80 | Answers supported by context |
| Answer Relevancy | >0.85 | Directly addresses question |
| Context Precision | >0.70 | Relevant chunks ranked high |
| Context Recall | >0.70 | All relevant info retrieved |

### Improving Scores

- **Low Faithfulness**: Adjust prompt to emphasize grounding
- **Low Relevancy**: Improve query understanding/reformulation
- **Low Precision**: Tune retrieval threshold
- **Low Recall**: Increase TOP_K or improve chunking

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
- **RAGAS** - Comprehensive RAG evaluation framework
- **Dataset** - `atitaarora/qdrant_doc` from Hugging Face

## Support

For issues and questions:
- Check troubleshooting section above
- Review configuration in `.env`
- Check Qdrant and Replicate documentation
- Open an issue on GitHub

## Roadmap

- [ ] Add support for multi-query retrieval
- [ ] Implement hybrid search (vector + keyword)
- [ ] Add conversation memory
- [ ] Support for multiple collections
- [ ] Web UI interface
- [ ] Docker compose setup
- [ ] Production deployment guide
