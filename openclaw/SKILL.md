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

## Quick Start

After installing QRAS and its dependencies, set up `qmem`:

```bash
cd ~/.openclaw/workspace

# Copy qmem wrapper to workspace root (if not already there)
cp qras/openclaw/qmem ./qmem
chmod +x ./qmem

# Create config from template
cp .qmem.conf.example .qmem.conf

# Edit config with your settings
nano .qmem.conf
```

**Config file (`.qmem.conf`):**

```bash
# Required
QRAS_DIR=/home/user/.openclaw/workspace/qras
COLLECTION=oc_memory
OLLAMA_HOST=localhost:11434

# Optional (defaults shown)
MIN_SCORE=0.22
LIMIT=3
MAX_CONTENT=600
```

**Test it:**

```bash
./qmem "test query"
```

## Installation (Full)

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- [Qdrant](https://qdrant.tech) vector database

### Setup Steps

1. **Clone QRAS to workspace:**

   ```bash
   cd ~/.openclaw/workspace
   git clone https://github.com/user/qras.git  # or copy from skill
   ```

2. **Install Python dependencies:**

   ```bash
   cd qras
   pip install -r requirements.txt
   ```

3. **Start Qdrant (using Docker):**

   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 \
       -v $(pwd)/qdrant_storage:/qdrant/storage:z \
       qdrant/qdrant
   ```

4. **Start Ollama and pull embedding model:**

   ```bash
   ollama serve  # if not already running
   ollama pull bge-m3:567m
   ```

5. **Make CLI executable:**

   ```bash
   chmod +x ./qras
   ```

6. **Set up qmem wrapper** (see Quick Start above)

7. **Create initial index:**

   ```bash
   ./qras index --input-path ~/.openclaw/workspace/memory --collection oc_memory --file-type markdown
   ./qras index --input-path ~/.openclaw/workspace/MEMORY.md --collection oc_memory
   ```

## Commands

### Search — Use `qmem`

```bash
# Basic search
./qmem "search query"

# With options
./qmem "search query" --limit 5
./qmem "search query" --collection other_collection

# Help
./qmem --help
```

**Always use `qmem` for memory recall.** It reads config and handles optimal parameters automatically.

**Query style:** Use natural language / intent-based queries.

- ✅ "when is the user's wife's birthday"
- ✅ "what did we decide about the invoice API"

### Direct QRAS Commands (Advanced)

For operations beyond search, use `qras` directly:

```bash
cd ~/.openclaw/workspace/qras

# Index single file
./qras index --input-path /path/to/file.md --collection oc_memory

# Delete file from index
./qras index --delete "filename.md" --collection oc_memory

# Full re-index
./qras index --input-path ~/.openclaw/workspace/memory --collection oc_memory --file-type markdown --recreate
```

## Workflow

1. **Always use `qmem` first** for memory recall
2. If no results or error → fallback to built-in `memory_search`
3. Use `memory_get` to pull specific lines after finding matches
4. **After updating any memory file** (`memory/*.md`, `MEMORY.md`), re-index it:
   ```bash
   cd ~/.openclaw/workspace/qras && ./qras index --input-path /path/to/file.md --collection oc_memory
   ```

## ⚠️ Index Scope (STRICT)

**ONLY index memory files:**
- `workspace/memory/*.md` — daily logs
- `workspace/MEMORY.md` — long-term memory

**DO NOT index:**
- Skills, tools, or other workspace files
- Project documentation outside memory
- Anything not explicitly memory-related

This keeps the collection focused on recall tasks and avoids polluting search results with unrelated content.

## Configuration Reference

### `.qmem.conf` Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `QRAS_DIR` | ✅ | - | Path to QRAS installation |
| `COLLECTION` | ✅ | - | Qdrant collection name |
| `OLLAMA_HOST` | ✅ | - | Ollama server (host:port) |
| `MIN_SCORE` | ❌ | 0.22 | Minimum relevance score |
| `LIMIT` | ❌ | 3 | Max results returned |
| `MAX_CONTENT` | ❌ | 600 | Max content chars per result |

### Environment Override

You can override config location:

```bash
QMEM_CONFIG=/path/to/.qmem.conf ./qmem "query"
```

## Local Config (TOOLS.md)

After setup, add reference to your `TOOLS.md`:

```markdown
## QRAS - Memory Search (Primary)

→ See skill: `~/.openclaw/skills/qras/SKILL.md`

### Quick Commands

```bash
# Search — ALWAYS use qmem for memory recall
./qmem "search query"

# Index after updating memory
cd qras && ./qras index --input-path /path/to/file.md --collection oc_memory
```
```

## Agent Setup (Important!)

After installing QRAS, **update your `AGENTS.md`** to enforce QRAS-first behavior:

```markdown
### 🔍 Memory Recall Rule

**QRAS first, `memory_search` fallback.** When recalling anything (siapa, kapan, apa), always use the QRAS skill first. Only fall back to built-in `memory_search` if QRAS returns no results or errors.

### ⚠️ Index After Writing Memory

Every time you create/update a memory file (`memory/*.md`, `MEMORY.md`), re-index it:

```bash
cd ~/.openclaw/workspace/qras && ./qras index --input-path <path-to-file> --collection oc_memory
```
```

This ensures you don't reflexively use built-in `memory_search` when QRAS is available.
