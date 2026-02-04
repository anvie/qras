# Qdrant RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that combines semantic search with LLM generation, featuring both CLI tools and a web interface.

## Features

- **CLI Interface**: Use CLI for indexing, searching, and chatting
- **Semantic Vector Search**: Search documents using semantic similarity
- **RAG Chat System**: Context-aware chat powered by LLMs with retrieved documents
- **Multiple Data Sources**: Index from JSON files or markdown directories
- **Hybrid Search**: Combines vector similarity with keyword matching for better relevance
- **Incremental Indexing**: Update single files without re-indexing everything
- **LLM-Optimized Output**: Token-efficient output format designed for AI agents
- **Interactive Modes**: Both search and chat interfaces with rich commands
- **Streaming Responses**: Real-time streaming of LLM responses
- **Web Interface**: FastAPI backend with Svelte frontend.

## Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- [Qdrant](https://qdrant.tech) vector database running

### Installation

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start required services:**

   ```bash
   # Start Qdrant (using Docker)
   docker run -p 6333:6333 -p 6334:6334 \
       -v $(pwd)/qdrant_storage:/qdrant/storage:z \
       qdrant/qdrant

   # Start Ollama and pull required models
   ollama serve
   ollama pull bge-m3:567m
   ollama pull llama3
   ```

3. **Make the CLI executable:**
   ```bash
   chmod +x ./qras
   ```

## CLI Usage

### Unified CLI Interface

Use the `./qras` script as your main entry point:

```bash
# Show help
./qras help

# Search documents
./qras query "machine learning algorithms"
./qras query "what is neural network" --hybrid --limit 5

# Index documents
./qras index --input-path ./documents --collection docs
./qras index --input-path ./articles.json --collection articles --recreate

# Interactive chat
./qras chat --interactive
./qras chat "What is machine learning?" --chat-model llama3

# Start web interface (backend + frontend)
./qras web
```

### Individual Commands

#### Search (`query`)

Search your indexed documents with semantic or hybrid search:

```bash
# Basic search (LLM-optimized output by default)
./qras query "machine learning algorithms"

# Hybrid search (vector + keyword)
./qras query "neural networks" --hybrid --limit 5

# Verbose mode (detailed output with emojis and timing)
./qras query "neural networks" --hybrid --verbose

# Save results to file
./qras query "deep learning" --output-format json --output results.json

# Interactive search mode
./qras query --interactive

# Show specific article
./qras query --article-id 123 --collection articles
```

**Key Options:**

- `--hybrid`: Use hybrid search (vector + keyword matching)
- `--limit`: Maximum number of results (default: 10)
- `--output-format`: `llm` (default), `verbose`, `compact`, or `json`
- `--verbose`: Show debug logs and detailed output with emojis
- `--max-content`: Max characters per result (default: 500)
- `--collection`: Collection to search (default: docs)
- `--interactive`: Interactive search mode

#### Index (`index`)

Index documents into the vector database:

```bash
# Index a directory of documents
./qras index --input-path ./documents --collection docs

# Index a single file (incremental update)
./qras index --input-path ./notes/meeting.md --collection docs

# Delete chunks for a specific file
./qras index --delete "meeting.md" --collection docs

# Index JSON files
./qras index --input-path ./articles.json --collection articles

# Recreate collection from scratch
./qras index --input-path ./data --recreate

# Custom chunking settings
./qras index --input-path ./docs --chunk-size 200 --chunk-overlap 50

# Show collection info
./qras index --info
./qras index --info --collection docs
```

**Key Options:**

- `--input-path`: Path to documents directory or single file
- `--delete`: Delete all chunks for a specific source/file
- `--collection`: Collection name (default: docs)
- `--recreate`: Delete and recreate collection
- `--chunk-size`: Words per chunk (default: 150)
- `--file-type`: `auto`, `json`, or `markdown`

**Note:** Single file indexing uses deterministic chunk IDs, enabling proper incremental updates without duplicates.

#### Chat (`chat`)

Interactive RAG chat combining search with LLM generation:

```bash
# Interactive chat mode
./qras chat --interactive

# Single question
./qras chat "What is machine learning?"

# Use different models
./qras chat --interactive --embedding-model bge-m3:567m --chat-model llama3

# Hybrid search in chat
./qras chat --interactive --hybrid --search-limit 3
```

**Key Options:**

- `--interactive`: Start interactive chat mode
- `--embedding-model`: Model for search embeddings
- `--chat-model`: LLM for response generation
- `--search-limit`: Max sources to retrieve (default: 5)
- `--hybrid`: Use hybrid search

#### Web Interface (`web`)

Start both backend and frontend servers simultaneously:

```bash
# Start web interface (backend + frontend)
./qras web
```

**Output:**

```
Starting Qdrant RAG Web Services...
====================================

Starting backend server on http://localhost:8000...
Starting frontend server on http://localhost:5173...

Web services are starting up...
Backend API: http://localhost:8000
Frontend UI: http://localhost:5173

Press Ctrl+C to stop all services
```

**Requirements:**

- Node.js and npm installed
- Backend dependencies installed (`pip install -r web/backend/app/requirements.txt`)
- Frontend dependencies installed (`cd web/frontend && npm install`)

## Web Interface Details

### Quick Start (Recommended)

Start both backend and frontend with a single command:

```bash
# Start both web backend and frontend simultaneously
./qras web
```

- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:5173
- **API Docs**: http://localhost:8000/api/v1/docs

Press `Ctrl+C` to stop both services.

### Manual Setup (Alternative)

If you prefer to run services separately:

#### Backend (FastAPI)

```bash
cd web/backend/app
python main.py
```

#### Frontend (Svelte)

```bash
cd web/frontend
npm install
npm run dev
```

## Configuration

### Environment Variables

Set environment variables to customize behavior:

```bash
# Database settings
export QDRANT_URL="http://localhost:6333"
export QDRANT_TIMEOUT=30

# Embedding settings
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_MODEL="embeddinggemma:latest"
export OLLAMA_TIMEOUT=120

# Indexing settings
export INDEXING_CHUNK_SIZE=150
export INDEXING_CHUNK_OVERLAP=30

# Search settings
export SEARCH_DEFAULT_LIMIT=10
export SEARCH_ENABLE_HYBRID=true
```

### Configuration File

Create a JSON configuration file:

```json
{
  "database": {
    "url": "http://localhost:6333",
    "timeout": 30
  },
  "embedding": {
    "url": "http://localhost:11434",
    "model": "embeddinggemma:latest",
    "timeout": 120
  },
  "indexing": {
    "chunk_size": 150,
    "chunk_overlap": 30
  }
}
```

Use with: `./qras query "test" --config-file config.json`

## Supported Models

### Embedding Models

- `embeddinggemma:latest` (768 dims) - Fast and efficient
- `bge-m3:567m` (1024 dims) - Multilingual, high quality
- `bge-large:latest` (1024 dims) - Large model for quality
- `all-minilm-l6-v2` (384 dims) - Compact and fast

### Chat Models

- `llama3` - General purpose conversational AI
- `codellama` - Code-focused responses
- `mistral` - Efficient and capable
- `gemma` - Google's efficient model
- Any Ollama-compatible model

## Examples

### Complete Workflow

1. **Index some documents:**

   ```bash
   ./qras index --input-path ./my-documents --collection knowledge
   ```

2. **Search the indexed content:**

   ```bash
   ./qras query "artificial intelligence" --collection knowledge --hybrid
   ```

3. **Chat with your documents:**

   ```bash
   ./qras chat --interactive --collection knowledge
   ```

4. **Use the web interface:**
   ```bash
   ./qras web
   ```

### Advanced Usage

```bash
# Index with custom settings
./qras index \
  --input-path ./documents \
  --collection docs \
  --model bge-m3:567m \
  --chunk-size 200 \
  --chunk-overlap 40 \
  --workers 8

# Search with specific parameters
./qras query \
  "machine learning algorithms" \
  --collection docs \
  --hybrid \
  --limit 15 \
  --min-score 0.1 \
  --output-format json

# Chat with custom prompts
./qras chat \
  --interactive \
  --collection docs \
  --embedding-model bge-m3:567m \
  --chat-model llama3 \
  --system-prompt "You are a technical expert. Provide detailed explanations."
```

## Development

### Project Architecture

The system is built with a clean separation between:

- **Shared Library (`lib/`)**: Common functionality used by both CLI and web
- **CLI Tools (`cli/`)**: Command-line interfaces for different operations
- **Web Interface (`web/`)**: FastAPI backend + Svelte frontend

### Common Issues

1. **Import errors**: Make sure you're using `./qras` (not individual Python scripts)
2. **Permission denied**: Run `chmod +x ./qras` to make the script executable
3. **Connection errors**: Verify Qdrant and Ollama are running
4. **Model not found**: Pull the required model with `ollama pull <model-name>`
5. **Empty results**: Check if documents are properly indexed

### Debug Mode

Run with verbose output:

```bash
LOG_LEVEL=DEBUG ./qras query "test"
```

### Vector Dimension Mismatch

If you get a "Vector dimension error", ensure the embedding model used for querying matches the one used for indexing:

```bash
# Check collection info
./qras query --interactive
# Then type: stats

# Re-index with correct model if needed
./qras index --input-path ./docs --recreate --model <correct-model>
```

### Connection Issues

**Ollama connection refused:**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

**Qdrant connection refused:**

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Start Qdrant if needed
docker run -p 6333:6333 qdrant/qdrant
```

## Performance Tips

1. **Use hybrid search** for better relevance:

   ```bash
   ./qras query "your query" --hybrid
   ```

2. **Optimize chunk size** based on your content:

   - Shorter chunks (100-150 words) for precise retrieval
   - Longer chunks (200-300 words) for more context

3. **Adjust search parameters** for RAG:

   - Lower limit (3-5) for focused responses
   - Higher limit (7-10) for comprehensive answers

4. **Use streaming** for better UX:
   ```bash
   ./qras chat --interactive  # Streaming enabled by default
   ```

## Acknowledgments

- [Qdrant](https://qdrant.tech/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Svelte](https://svelte.dev/) - Frontend framework

---

**Author:** Robin Syihab
