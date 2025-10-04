# codebase-search

Index your entire codebase to empower intelligent coding agents with precise search capabilities. This solution, leveraging pgvector, sentence-transformers, and the MCP Python SDK, offers a basic setup for a quick start.

## Features

- Semantic codebase search using vector embeddings
- Git-tracked files only with smart filtering
- Support for all text-based languages and file types
- MCP server for AI assistant integration
- Content hashing to skip unchanged files during re-indexing
- Intelligent chunking with overlapping sections
- Fast vector search with cosine similarity

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+

### Setup

```bash
# 1. Start database and MCP server
docker-compose up -d --build

# 2. Install dependencies
pip install -r requirements.txt

# 3. Index codebase (one-time, 5-10 minutes)
python codebase_indexer.py
```

## Usage

### Standalone Search

```bash
# Interactive search
python search_codebase.py

# Direct queries
python search_codebase.py "Angular component" -n 3
python search_codebase.py "HTTP service API" -n 5
python search_codebase.py "authentication guard" -n 2
```

### MCP Server Integration

The system includes an MCP (Model Context Protocol) server for AI assistant integration:

```bash
docker-compose up -d --build
```

**Available MCP Tools:**

- `search_codebase(query, limit)` - Semantic codebase search
- `search_by_file_type(extension, query, limit)` - Search specific file types
- `get_codebase_stats()` - Get indexing statistics

**Client Configuration:**

```json
{
	"mcpServers": {
		"codebase-search": {
			"type": "http",
			"url": "http://localhost:8111/mcp"
		}
	}
}
```

## Search Examples

### Functional Searches

```
"authentication service"     → Find auth-related services
"HTTP interceptor"           → Find HTTP interceptors
"Angular component routing"  → Find routing components
"database migration"         → Find migration scripts
```

### Architectural Searches

```
"service injection"          → Find dependency injection
"guard implementation"       → Find route guards
"error handling"             → Find error handling code
"form validation"            → Find form validation logic
```

### Domain-Specific Searches

```
"about this app"             → Find the about this app pages
"search functionality"       → Find search implementations
"user authentication"        → Find user auth workflows
```

## What Gets Indexed

- **Git-tracked files only**: Uses `git ls-files` to discover files
- **Text files only**: Automatically detects and skips binary files
- **All languages**: TypeScript, JavaScript, HTML, CSS, JSON, Markdown, etc.
- **Smart filtering**: Excludes images, fonts, archives automatically
- **Respects .gitignore**: Only indexes what git tracks

## Architecture

### Database Schema

Makes use of the `pgvector` extension for vector search.

```sql
CREATE TABLE codebase_chunks
(
	--- embedding_dim=384
	embedding halfvec(384)
);
```

### Embedding Model

Configurable via `EMBEDDING_MODEL` (default: all-MiniLM-L6-v2 with `embedding_dim=384`)

## Configuration

### Environment Variables (.env)

DB connection and model configuration are stored in `.env`.

```bash
DB_HOST=localhost
DB_PORT=5432
POSTGRES_DB=codebase-search
POSTGRES_USER=dev
POSTGRES_PASSWORD=dev

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
#EMBEDDING_MODEL=nomic-ai/nomic-embed-code

# Chunking Configuration
TOKENIZER_ENCODING=cl100k_base
MAX_TOKENS_PER_CHUNK=500
CHUNK_OVERLAP=50

# Where to find the codebase on the host machine
HOST_PROJECT_ROOT=.
```

### Configuration Notes

- **Auto-detection**: Embedding dimensions are automatically detected from the selected model
- **Chunking**: Token limits and overlap are configurable via environment variables
- **Tokenizer**: Uses OpenAI's cl100k_base tokenizer by default (works well across models)
- **Model flexibility**: Change `EMBEDDING_MODEL` to switch between different Sentence Transformer models

## Re-indexing

After code changes:

```bash
python codebase_indexer.py
```

The system uses content hashing to skip unchanged files for efficiency.

## Use Cases for AI Assistants

The MCP server enables AI assistants to:

1. **Code Analysis**: Understand existing code patterns and architectures
2. **Bug Investigation**: Find relevant code sections for debugging
3. **Feature Development**: Locate similar implementations for reference
4. **Code Review**: Search for best practices and consistency patterns
5. **Documentation**: Find examples and usage patterns
6. **Refactoring**: Identify code that needs to be updated together

## Troubleshooting

### Common Issues

**Connection Errors**: Check Docker with `docker compose up -d --build --force-recreate`

## Files

### Core Scripts

- `codebase_indexer.py` - The main indexing script
- `search_codebase.py` - Command-line search interface
- `mcp_server.py` - MCP server for AI assistant integration

### Configuration

- `Dockerfile` - Container image for the MCP server
- `docker-compose.yml` - PostgreSQL database with pgvector
- `requirements.txt` - Python dependencies
- `.env` - Database configuration
- `AGENTS.md` - Instructions for AI assistant integration
