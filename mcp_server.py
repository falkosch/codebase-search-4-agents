#!/usr/bin/env python3
import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP

from codebase_indexer import CodebaseIndexer

logger = logging.getLogger(__name__)
mcp = FastMCP(name='codebase-search-mcp', log_level='DEBUG', host=os.getenv('FASTMCP_HOST', '127.0.0.1'), port=8111)
indexer: Optional[CodebaseIndexer] = None


@dataclass
class SearchResult:
  file_path: str
  language: str
  similarity: float
  chunk_index: int
  chunk_content: str


@mcp.tool()
def search_codebase(query: str, limit: int = 5) -> List[Dict[str, Any]]:
  """
  Search the codebase using semantic similarity.

  Args:
      query: The search query (e.g., "Angular component", "authentication guard")
      limit: Maximum number of results to return (default: 5, max: 20)

  Returns:
      List of search results with file paths, similarity scores, and code snippets
  """
  try:
    # Validate inputs
    limit = max(1, min(limit, 20))

    results = indexer.search_code(query, limit=max(1, min(limit, 20)))

    # Convert to structured format
    search_results = []
    for result in results:
      chunk_content = result['chunk_content']
      search_results.append({
        'file_path': result['file_path'],
        'language': result['language'],
        'similarity': round(result['similarity'], 3),
        'chunk_index': result['chunk_index'],
        'content_preview': chunk_content[:300] + '...' if len(chunk_content) > 300 else chunk_content,
        'content_lines': len(chunk_content.split('\n'))
      })
    return search_results
  except Exception as e:
    logger.error('Search failed: %s', e)
    return [{'error': f'Search failed: {str(e)}'}]


@mcp.tool()
def search_by_file_type(file_extension: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
  """
  Search within files of a specific type/extension.

  Args:
      file_extension: File extension to filter by (e.g., "ts", "html", "scss")
      query: Search query
      limit: Maximum number of results (default: 5, max: 20)

  Returns:
      List of search results filtered by file type
  """
  try:
    # Validate inputs
    limit = max(1, min(limit, 20))
    file_extension = file_extension.lower().lstrip('.')

    # Perform general search first
    all_results = search_codebase(query, limit * 3)  # Get more results to filter

    # Filter by file extension
    filtered_results = []
    for result in all_results:
      if 'error' in result:
        continue

      file_path = result['file_path']
      if file_path.lower().endswith(f'.{file_extension}'):
        filtered_results.append(result)

      if len(filtered_results) >= limit:
        break

    return filtered_results

  except Exception as e:
    logger.error('File type search failed: %s', e)
    return [{'error': f'Search failed: {str(e)}'}]


@mcp.tool()
def get_codebase_stats() -> Dict[str, Any]:
  """
  Get statistics about the indexed codebase.

  Returns:
      Dictionary with codebase statistics including file counts and index info
  """
  try:
    # Query database for statistics
    cursor = indexer.cursor

    # Total chunks
    cursor.execute('SELECT COUNT(*) as count FROM codebase_chunks')
    result = cursor.fetchone()
    total_chunks = result['count'] if result else 0

    # Total files
    cursor.execute('SELECT COUNT(DISTINCT file_path) as count FROM codebase_chunks')
    result = cursor.fetchone()
    total_files = result['count'] if result else 0

    # Files by language
    cursor.execute("""
                   SELECT language, COUNT(DISTINCT file_path) as file_count
                   FROM codebase_chunks
                   WHERE language IS NOT NULL
                   GROUP BY language
                   ORDER BY file_count DESC
                   LIMIT 10
                   """)
    languages = {row['language']: row['file_count'] for row in cursor.fetchall()}

    # Most recent indexing
    cursor.execute("SELECT MAX(created_at) as last_indexed FROM codebase_chunks")
    result = cursor.fetchone()
    last_indexed = result['last_indexed'] if result else None

    return {
      'total_files': total_files,
      'total_chunks': total_chunks,
      'languages': languages,
      'last_indexed': str(last_indexed) if last_indexed else None,
      'average_chunks_per_file': round(total_chunks / total_files, 2) if total_files > 0 else 0
    }

  except Exception as e:
    logger.error('Failed to get stats: %s', e)
    return {'error': f'Failed to get statistics: {str(e)}'}


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='MCP server for searching the codebase',
    epilog='Example: python mcp_server.py --project=/path/to/project'
  )
  parser.add_argument(
    '--project',
    type=Path,
    default=Path(os.getcwd()),
    help='Project root directory',
  )
  args = parser.parse_args()
  logger.info(f'mcp_server - Project root: %s', args.project)
  indexer = CodebaseIndexer(args.project)
  indexer.connect_db()
  try:
    mcp.run('streamable-http')
  except KeyboardInterrupt:
    logger.info('Sighup received, shutting down...')
  finally:
    indexer.disconnect_db()
