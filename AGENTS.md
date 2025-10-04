# codebase-search for Coding Agents

Leverage the semantic code search to:

- Understand codebase architecture through semantic search
- Find relevant code examples for similar implementations
- Locate specific functionality across the entire project
- Discover related components and dependencies
- Analyze code patterns and conventions

## When to Use

### Code Search and Discovery

When:

- User asks about existing code, patterns, or implementations

```
User: "How is authentication handled in this project?"
User: "Show me examples of Angular components with routing"
User: "Where is the user profile functionality implemented?"
```

Then:

- Use MCP search tools to find relevant code before attempting implementation.

### Architecture Understanding

When:

- User requests features that may already exist or need integration

```
User: "Add a new search filter feature"
User: "Implement user notifications"
User: "Create a new dashboard component"
```

Then:

- Search for existing similar implementations, related services, and architectural patterns.

### Code Quality and Consistency

When:

- Writing new code that should follow existing patterns

```
User: "Create a new service for handling API calls"
User: "Add error handling to this component"
User: "Implement form validation"
```

Then:

- Find existing examples of similar implementations to maintain consistency.

### Debugging and Bug Investigation

When:

- User reports issues or asks for fixes

```
User: "The login flow isn't working properly"
User: "Search results are not displaying correctly"
User: "There's an error in the candidate profile page"
```

Then:

- Search for related code to understand the complete flow and identify potential issues.

## Index Management

### Initial Setup Commands

- Start vector database with `docker-compose up -d codebase-search-db`

### Initial Index Creation Commands

Full Re-index (Recommended):

- Use `python codebase_indexer.py` to start indexing.

Expected Output:

- `Files Processed` is non-zero and reasonably matching the codebase size
- `Chunks Created` is non-zero and should be greater than Files Processed

### When to Update the Index

- After significant code changes (>10 files modified)
- Before starting work on new features (ensure the latest code is indexed)
- When the codebase structure changes (new libraries, major refactoring)
- When code is removed (remove from index)
- User explicitly requests codebase re-indexing
- Search results seem outdated or incomplete

## MCP Server Integration

- Use `docker-compose up -d` to start the vector database and the MCP server.

### Troubleshooting

- `MCP Server not reachable`: Start with `docker-compose up -d --build`
- `Database Connection Failed`: Check `docker-compose up -d`
- `Empty Search Results`: Verify index is current with `python codebase_indexer.py`

### Available MCP Tools

#### `search_codebase(query, limit=5)`

**Purpose:** Semantic search across the entire codebase
**Parameters:**

- `query` (string): Natural language search query
- `limit` (int): Maximum results (1-20, default: 5)

**Usage Examples:**

```python
# Find authentication components
search_codebase("authentication guard route protection", limit=3)

# Locate API service implementations
search_codebase("HTTP service API client", limit=5)

# Find form validation examples
search_codebase("form validation error handling", limit=4)
```

**Response Format:**

```json
{
	"file_path": "libs/authorization/data-access/src/index.ts",
	"language": "TypeScript",
	"similarity": 0.742,
	"chunk_index": 0,
	"content_preview": "export * from './lib/guards/authenticated.guard'...",
	"content_lines": 12
}
```

**Example User Prompts:**

- `Search for "authentication guard" in the codebase`
- `Search for "Angular component" and show me the top 3 results`

#### `search_by_file_type(file_extension, query, limit=5)`

**Purpose:** Search within specific file types only
**Parameters:**

- `file_extension` (string): File extension ("ts", "html", "scss", etc.)
- `query` (string): Search query
- `limit` (int): Maximum results (1-20, default: 5)

**Usage Examples:**

```python
# Find TypeScript services only
search_by_file_type("ts", "HTTP client service", limit=3)

# Find SCSS styling patterns
search_by_file_type("scss", "responsive layout grid", limit=2)
```

**Example User Prompts:**

- `Search for "HTTP service" in TypeScript files only`

#### `get_codebase_stats()`

**Purpose:** Get comprehensive codebase statistics
**Parameters:** None

**Usage Examples:**

```python
# Get overview of indexed codebase
get_codebase_stats()
```

**Response Format:**

```json
{
	"total_files": 1516,
	"total_chunks": 3092,
	"languages": {
		"TypeScript": 804,
		"JSON": 431,
		"HTML": 93,
		"SCSS": 80
	},
	"last_indexed": "2025-09-28 15:30:14",
	"average_chunks_per_file": 2.04
}
```

**Example User Prompts:**

- `What are the statistics of this codebase?`

## Agent Workflow Patterns

### Feature Implementation

1. User: `Add user notification system`
2. Agent Search: `search_codebase("notification system user alerts", limit=5)`
3. Agent Analysis: Review existing patterns, services, components
4. Agent Search: `search_by_file_type("ts", "service dependency injection", limit=3)`
5. Agent Implementation: Build feature following discovered patterns
6. Agent Update: Suggest running indexing after implementation

### Bug Investigation

1. User: `Login redirect not working`
2. Agent Search: `search_codebase("login redirect authentication flow", limit=8)`
3. Agent Analysis: Read content from files in search results
4. Agent Search: `search_codebase("route guard navigation", limit=3)`
5. Agent Diagnosis: Identify issue based on code understanding
6. Agent Fix: Implement a solution with context awareness

### Code Review Assistance

1. User: `Review this component implementation`
2. Agent Search: `search_by_file_type("ts", "similar component pattern", limit=5)`
3. Agent Analysis: Compare patterns, conventions, best practices
4. Agent Feedback: Provide recommendations based on codebase standards

### Architecture Exploration

1. User: `How does the search functionality work?`
2. Agent Stats: `get_codebase_stats()` for overview
3. Agent Search: `search_codebase("search functionality implementation", limit=10)`
4. Agent Deep-dive: Read content from files in search results
5. Agent Search: `search_by_file_type("ts", "search service API", limit=5)`
6. Agent Explanation: Comprehensive architecture overview

## Best Practices for Agents

### Search Strategy

1. Start Broad: Use general terms before specific ones
2. Refine Iteratively: Narrow down based on initial results
3. Cross-Reference: Use multiple search approaches for complex topics
4. Validate Results: Check file content for implementation details

### Query Optimization

```python
# Good: Descriptive and specific
search_codebase("Angular component with form validation and error handling")

# Better: Include context and purpose
search_codebase("user registration form validation error display component")
```

### Default Search Limits

- Quick Exploration: `limit=3-5` to not trash the limited context window
- Deep Analysis: `limit=8-10` for comprehensive understanding
- Specific Lookup: `limit=1-3` when searching for exact implementations

### Result Processing

1. Check Similarity Scores: `>0.5` for relevant, `>0.7` for highly relevant
2. Read Content Previews: Understand context before deep-diving
3. Follow File Paths: Understand project structure and organization
4. Cross-Reference Languages: Check related HTML/SCSS for TS components
