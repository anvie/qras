---
name: qras
description: Semantic memory search via QRAS (Qdrant RAG System). Use as PRIMARY method for recalling workspace files, notes, decisions, people, preferences, and prior work. Triggers on any memory recall task before falling back to built-in memory_search.
---

# QRAS - Semantic Memory Search

Local RAG system using Qdrant + Ollama embeddings.

## When to Use

Use QRAS as **primary method** for:

- Recalling past conversations, decisions, or events
- Finding user preferences, birthdays, names, relationships
- Searching workspace notes, documentation, or project files
- Any question about "what did we...", "when did...", "who is..."
- Looking up prior work, todos, or context from memory files

**Fallback:** If QRAS returns no results or errors, use built-in `memory_search`.

## Installation

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- [Qdrant](https://qdrant.tech) vector database

### Setup Steps

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start Qdrant (using Docker):**

   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 \
       -v $(pwd)/qdrant_storage:/qdrant/storage:z \
       qdrant/qdrant
   ```

3. **Start Ollama and pull embedding model:**

   ```bash
   ollama serve  # if not already running
   ollama pull bge-m3:567m
   ```

4. **Make CLI executable:**

   ```bash
   chmod +x ./qras
   ```

5. **Configure environment (optional):**

   ```bash
   cp .env.example .env
   # Edit .env with your Qdrant/Ollama host settings
   ```

## Commands

### Search (Primary Use)

```bash
./qras query "<semantic query>" --collection <collection> --hybrid --max-content 600 --min-score 0.2 --limit 3
```

**Query style:** Use natural language / intent-based queries.

- ✅ "when is the user's wife's birthday"
- ✅ "what did we decide about the invoice API"

### Index Single File

```bash
./qras index --input-path /path/to/file.md --collection <collection>
```

### Delete File from Index

```bash
./qras index --delete "filename.md" --collection <collection>
```

### Full Re-index

```bash
./qras index --input-path /path/to/docs --collection <collection> --file-type markdown --recreate
```

## Workflow

1. **Always try QRAS first** for memory recall
2. If no results or error → fallback to built-in `memory_search`
3. Use `memory_get` to pull specific lines after finding matches

## Optimal Parameters

| Param | Value | Reason |
|-------|-------|--------|
| `--hybrid` | always | Combines vector + keyword for accuracy |
| `--max-content` | 600 | Balance context vs token cost |
| `--min-score` | 0.2 | Filter low-relevance noise |
| `--limit` | 3 | Usually sufficient; increase if needed |

## Local Config

After installation, add your local settings to `TOOLS.md`:

```markdown
## QRAS - Memory Search

Path: /path/to/qras
Collection: your-collection-name
Ollama: localhost:11434
Model: bge-m3:567m
```
