# Cohere Toolkit Integration Guide

This guide explains how to integrate your Qdrant RAG pipeline with the Cohere Toolkit frontend.

## Overview

The integration allows users to query your Qdrant vector database through the Cohere Toolkit's chat interface. The RAG tool appears as a selectable data source in the UI.

## Architecture

```
User (Cohere UI)
    ↓
Cohere Toolkit Backend (FastAPI)
    ↓
QdrantRAGTool (this integration)
    ↓
RAG Pipeline (your code)
    ↓
Qdrant Vector Database
```

## Prerequisites

1. **Cohere Toolkit** installed and running
   - Clone from: https://github.com/parzival1l/cohere-toolkit/
   - Follow their setup instructions

2. **RAG Pipeline** (this repository) configured
   - Qdrant cluster URL and API key
   - Replicate API token
   - Collection name

3. **Indexed Data** in your Qdrant collection
   - Run `python main.py index` from this repository

## Installation Steps

### 1. Copy the Tool to Cohere Toolkit

```bash
# Navigate to your Cohere Toolkit directory
cd /path/to/cohere-toolkit

# Copy the RAG tool
cp /home/user/RAG-exploration/cohere_integration/qdrant_rag_tool.py \
   src/community/tools/
```

### 2. Update Imports

Edit `src/community/tools/__init__.py`:

```python
# Add to imports
from community.tools.qdrant_rag_tool import QdrantRAGTool

# Add to __all__
__all__ = [
    # ... existing tools
    "QdrantRAGTool",
]
```

### 3. Register the Tool

Edit `src/community/config/tools.py`:

```python
# Add to imports
from community.tools import (
    # ... existing imports
    QdrantRAGTool,
)

# Add to enum
class CommunityTool(Enum):
    # ... existing tools
    Qdrant_RAG = QdrantRAGTool
```

### 4. Configure Environment Variables

Edit the Cohere Toolkit's `.env` file:

```bash
# ============================================
# QDRANT RAG TOOL CONFIGURATION
# ============================================

# Your Qdrant cluster URL (required)
QDRANT_URL=https://your-cluster.qdrant.io:6333

# Your Qdrant API key (required for cloud)
QDRANT_API_KEY=your_qdrant_api_key_here

# Your collection name (required)
COLLECTION_NAME=your_collection_name_here

# Replicate API token (required)
REPLICATE_API_TOKEN=your_replicate_token_here

# RAG Pipeline path (optional, defaults to /home/user/RAG-exploration)
RAG_PIPELINE_PATH=/home/user/RAG-exploration

# ============================================
# OPTIONAL: ADVANCED CONFIGURATION
# ============================================

# Embedding model (optional, uses default if not set)
EMBEDDING_MODEL=beautyyuyanli/multilingual-e5-large

# Retrieval settings (optional)
TOP_K_CANDIDATES=15
SCORE_THRESHOLD=0.7

# Generation settings (optional)
LLM_MODEL=meta/meta-llama-3-70b-instruct
TEMPERATURE=0.2
MAX_TOKENS=1000

# ============================================
# FEATURE FLAGS
# ============================================

# Enable community tools (required)
USE_COMMUNITY_FEATURES=true
```

### 5. Restart Cohere Toolkit

```bash
# Stop the services
docker-compose down

# Rebuild with new changes
docker-compose build

# Start services
docker-compose up -d

# Or for development:
make dev
```

## Testing the Integration

### 1. Verify Tool Registration

Check if the tool is available:

```bash
curl -X GET "http://localhost:8000/v1/tools" \
  -H "User-Id: test-user" | jq '.[] | select(.name=="qdrant_rag_retriever")'
```

Expected output:
```json
{
  "name": "qdrant_rag_retriever",
  "display_name": "Qdrant RAG Retriever",
  "description": "Retrieves relevant information from a Qdrant vector database...",
  "is_available": true,
  "category": "Data loader",
  "parameter_definitions": {
    "query": {
      "description": "The search query or question...",
      "type": "str",
      "required": true
    },
    "num_results": {
      "description": "Number of documents to retrieve...",
      "type": "int",
      "required": false
    }
  }
}
```

### 2. Test Chat API

Test with a simple query:

```bash
curl -X POST "http://localhost:8000/v1/chat" \
  -H "Content-Type: application/json" \
  -H "User-Id: test-user" \
  -d '{
    "message": "What is Qdrant?",
    "tools": [{"name": "qdrant_rag_retriever"}],
    "model": "command-r-plus"
  }' | jq
```

### 3. Test Streaming

Test with streaming response:

```bash
curl -N -X POST "http://localhost:8000/v1/chat-stream" \
  -H "Content-Type: application/json" \
  -H "User-Id: test-user" \
  -d '{
    "message": "Explain vector search",
    "tools": [{"name": "qdrant_rag_retriever"}]
  }'
```

### 4. Test in UI

1. Open browser to `http://localhost:4000`
2. Click on the tools/settings icon
3. Find and enable "Qdrant RAG Retriever"
4. Ask a question that should use your documentation
5. Verify:
   - Tool is called (you'll see "Searching..." or similar)
   - Results are returned with citations
   - Sources are properly attributed

## Troubleshooting

### Tool Not Appearing

**Problem**: Tool doesn't show up in `/v1/tools` endpoint

**Solutions**:
1. Check if `USE_COMMUNITY_FEATURES=true` is set
2. Verify imports in `__init__.py` are correct
3. Check enum registration in `tools.py`
4. Restart the backend service
5. Check logs: `docker-compose logs backend`

### Tool Shows as Unavailable

**Problem**: Tool appears but `is_available: false`

**Solutions**:
1. Verify environment variables are set:
   ```bash
   docker-compose exec backend env | grep QDRANT
   docker-compose exec backend env | grep REPLICATE
   ```
2. Check required variables:
   - `QDRANT_URL`
   - `REPLICATE_API_TOKEN`
3. Ensure values don't have quotes or extra spaces

### No Results Returned

**Problem**: Tool runs but returns no documents

**Solutions**:
1. Verify your Qdrant collection exists:
   ```bash
   curl "https://your-cluster.qdrant.io:6333/collections" \
     -H "api-key: your_api_key"
   ```
2. Check collection has documents
3. Try lowering `score_threshold` in tool parameters
4. Verify embeddings are working (test with `main.py`)

### Import Errors

**Problem**: `ModuleNotFoundError` when starting backend

**Solutions**:
1. Ensure RAG pipeline path is correct:
   ```bash
   RAG_PIPELINE_PATH=/home/user/RAG-exploration
   ```
2. Mount the volume in docker-compose.yml:
   ```yaml
   volumes:
     - /home/user/RAG-exploration:/rag-pipeline:ro
   ```
3. Update path in environment:
   ```bash
   RAG_PIPELINE_PATH=/rag-pipeline
   ```

### Connection Refused to Qdrant

**Problem**: Can't connect to Qdrant cluster

**Solutions**:
1. Verify URL format (should include protocol and port)
2. Check firewall/network access
3. For Qdrant Cloud, verify API key is correct
4. Test connection outside Docker:
   ```bash
   curl https://your-cluster.qdrant.io:6333/collections \
     -H "api-key: your_key"
   ```

## Advanced Configuration

### Custom Tool Parameters

You can modify tool behavior by editing `qdrant_rag_tool.py`:

```python
# Change default number of results
num_results = parameters.get("num_results", 10)  # Default to 10

# Add custom metadata fields
result["custom_field"] = metadata.get("custom_field")

# Add relevance score to results
result["relevance_score"] = doc.metadata.get("score")
```

### Adding Preamble for Better RAG

Create a specialized agent with custom instructions:

1. In Cohere Toolkit, navigate to agent creation
2. Set preamble:
   ```
   You are a helpful assistant specialized in answering questions about [your domain].
   Always use the Qdrant RAG Retriever tool to find relevant information before answering.
   Cite your sources and be precise. If information is not in the retrieved documents, say so.
   ```
3. Enable Qdrant RAG Retriever tool by default
4. Save as "Documentation Assistant"

### Performance Optimization

**Caching**: Add Redis caching to reduce repeated queries:

```python
import redis
cache = redis.Redis(host='localhost', port=6379)

async def call(self, parameters: dict, **kwargs):
    query = parameters.get("query")
    cache_key = f"rag:{query}:{num_results}"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Retrieve and cache
    results = pipeline.retrieve_documents(...)
    cache.setex(cache_key, 3600, json.dumps(results))
    return results
```

**Connection Pooling**: Reuse Qdrant client across requests (already implemented via singleton pattern)

## Docker Deployment

If deploying with Docker Compose, add RAG pipeline as a service:

```yaml
# In cohere-toolkit/docker-compose.yml
services:
  backend:
    environment:
      - QDRANT_URL=${QDRANT_URL}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
      - COLLECTION_NAME=${COLLECTION_NAME}
      - REPLICATE_API_TOKEN=${REPLICATE_API_TOKEN}
      - RAG_PIPELINE_PATH=/rag-pipeline
    volumes:
      - /home/user/RAG-exploration:/rag-pipeline:ro
```

## API Reference

### Tool Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | Yes | - | The search query or question |
| num_results | integer | No | 5 | Number of documents (1-20) |
| score_threshold | float | No | 0.7 | Minimum similarity score (0.0-1.0) |

### Response Format

Each document in the response contains:

```json
{
  "text": "The actual document content...",
  "title": "Document Title",
  "url": "https://source.url/path"
}
```

## Support

For issues:
1. Check Cohere Toolkit logs: `docker-compose logs -f backend`
2. Check RAG pipeline logs: `python main.py stats`
3. Test tool independently: `python cohere_integration/qdrant_rag_tool.py`
4. Verify Qdrant connection: `curl <QDRANT_URL>/collections`

## Next Steps

1. **Customize the tool** for your specific use case
2. **Add more metadata** to improve citations
3. **Create specialized agents** with custom preambles
4. **Implement caching** for better performance
5. **Add monitoring** and logging
6. **Write tests** in `src/community/tests/tools/test_qdrant_rag_tool.py`
