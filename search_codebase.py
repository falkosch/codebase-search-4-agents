#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from codebase_indexer import CodebaseIndexer

indexer: Optional[CodebaseIndexer] = None
logger = logging.getLogger(__name__)


def search_codebase(query: str, limit: int = 5):
  global indexer
  logger.info('Searching for: %s', query)
  results = indexer.search_code(query, limit=limit)
  if results:
    for i, result in enumerate(results, 1):
      logger.info('%s. %s', i, result['file_path'])
      logger.info('   Language: %s', result['language'])
      logger.info('   Similarity: %.3f', result['similarity'])
      logger.info('   Chunk: %s', result['chunk_index'])
      content_lines = result['chunk_content'].strip().split('\n')
      preview_lines = content_lines[:5]  # First 5 lines
      for line in preview_lines:
        logger.info('   | %s', line)
      if len(content_lines) > 5:
        logger.info('   | ... (%s more lines)', len(content_lines) - 5)
  else:
    logger.info('No results found.')


def main():
  global indexer
  parser = argparse.ArgumentParser(
    description='Search the indexed codebase',
    epilog='Example: python search_codebase.py "Angular component routing"'
  )
  parser.add_argument(
    "--project",
    type=Path,
    default=Path(os.getcwd()),
    help='Project root directory',
  )
  parser.add_argument(
    'query',
    help='Search query'
  )
  parser.add_argument(
    '-n', '--limit',
    type=int,
    default=5,
    help='Maximum number of results to show (default: 5)'
  )
  args = parser.parse_args()
  logger.info('search_codebase - Project root: %s', args.project)
  indexer = CodebaseIndexer(args.project)
  indexer.connect_db()
  search_codebase(args.query, args.limit)


def loop_input():
  global indexer
  indexer = CodebaseIndexer(Path(os.getcwd()))
  indexer.connect_db()
  while True:
    query = input('\nEnter search query (or "quit" to exit): ').strip()
    if query.lower() in ['quit', 'exit', 'q']:
      break
    if query:
      search_codebase(query)


if __name__ == '__main__':
  try:
    if len(sys.argv) == 1:
      loop_input()
    else:
      main()
  except KeyboardInterrupt:
    logger.info('Sighup received, shutting down...')
  finally:
    indexer.disconnect_db()
