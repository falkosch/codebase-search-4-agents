#!/usr/bin/env python3
import argparse
import hashlib
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Generator

import psycopg2
import tiktoken
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

LANGUAGE_MAP = {
  '.ts': 'TypeScript',
  '.tsx': 'TypeScript',
  '.js': 'JavaScript',
  '.jsx': 'JavaScript',
  '.py': 'Python',
  '.java': 'Java',
  '.kt': 'Kotlin',
  '.cs': 'C#',
  '.cpp': 'C++',
  '.cc': 'C++',
  '.cxx': 'C++',
  '.c': 'C',
  '.h': 'C/C++',
  '.hpp': 'C++',
  '.rs': 'Rust',
  '.go': 'Go',
  '.php': 'PHP',
  '.rb': 'Ruby',
  '.swift': 'Swift',
  '.scala': 'Scala',
  '.sql': 'SQL',
  '.html': 'HTML',
  '.htm': 'HTML',
  '.xml': 'XML',
  '.css': 'CSS',
  '.scss': 'SCSS',
  '.sass': 'SASS',
  '.less': 'LESS',
  '.json': 'JSON',
  '.yml': 'YAML',
  '.yaml': 'YAML',
  '.toml': 'TOML',
  '.md': 'Markdown',
  '.rst': 'reStructuredText',
  '.txt': 'Text',
  '.sh': 'Shell',
  '.bash': 'Shell',
  '.zsh': 'Shell',
  '.dockerfile': 'Dockerfile',
}


def _detect_language(file_path: Path) -> str:
  extension = file_path.suffix.lower()
  if file_path.name.lower() == 'dockerfile':
    return 'Dockerfile'
  return LANGUAGE_MAP.get(extension, 'Unknown')


def _compute_chunk_hash(file_path: str, chunk_index: int, content: str) -> str:
  hash_input = f'{file_path}:{chunk_index}:{content}'.encode('utf-8')
  return hashlib.sha256(hash_input).hexdigest()


def _should_index_file(file_path: Path) -> bool:
  try:
    if not file_path.exists() or not file_path.is_file() or file_path.is_symlink() or file_path.stat().st_size == 0:
      return False

    # Check for null bytes (common in binary files)
    with open(file_path, 'rb') as f:
      chunk = f.read(8000)
      if not chunk:  # Empty file
        return False
      if b'\x00' in chunk:
        return False

    return True
  except Exception as e:
    logger.debug('Error checking if %s is text: %s', file_path, e)
    return False


load_dotenv()

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  handlers=[
    logging.FileHandler(Path.cwd() / 'indexer.log'),
    logging.StreamHandler()
  ]
)
logger = logging.getLogger(__name__)


class CodebaseIndexer:
  def __init__(self, project_root: Path):
    self.connection = None
    self.cursor = None
    self.project_root = project_root.resolve()

    self._verify_git_repository()

    # Initialize embedding model with auto-detection
    model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    logger.info('Loading embedding model: %s', model_name)
    self.embedding_model = SentenceTransformer(model_name)
    self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
    logger.info('Auto-detected embedding dimension: %s', self.embedding_dim)

    # Initialize tokenizer and chunk parameters from the environment
    tokenizer_encoding = os.getenv('TOKENIZER_ENCODING', 'cl100k_base')
    self.max_tokens_per_chunk = int(os.getenv('MAX_TOKENS_PER_CHUNK', '500'))
    self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '50'))
    logger.info('Tokenizer: %s', tokenizer_encoding)
    logger.info('Max tokens per chunk: %s', self.max_tokens_per_chunk)
    logger.info('Chunk overlap: %s', self.chunk_overlap)
    self.tokenizer = tiktoken.get_encoding(tokenizer_encoding)

    self.stats = {
      'files_processed': 0,
      'chunks_created': 0,
      'files_skipped': 0,
      'errors': 0
    }

  def _verify_git_repository(self):
    result = subprocess.run(
      ['git', 'rev-parse', '--git-dir'],
      cwd=self.project_root,
      capture_output=True,
      text=True,
      timeout=10
    )
    if result.returncode != 0:
      logger.error('Git repository verification failed: %s %s', result.returncode, result.stderr)
      raise Exception('Project root is not a git repository')
    logger.info('Project root is a git repository')

  def _get_git_tracked_files(self) -> Set[Path]:
    try:
      result = subprocess.run(
        ['git', 'ls-files'],
        cwd=self.project_root,
        capture_output=True,
        text=True,
        timeout=30
      )
      if result.returncode != 0:
        raise Exception('Git ls-files failed: %s %s', result.returncode, result.stderr)

      tracked_files = set()
      for line in result.stdout.strip().split('\n'):
        if line:  # Skip empty lines
          file_path = self.project_root / line
          tracked_files.add(file_path)

      logger.info('Found %s git-tracked files', len(tracked_files))
      return tracked_files
    except Exception as e:
      logger.error('Failed to get git tracked files: %s', e)
      return set()

  def connect_db(self) -> bool:
    try:
      db_host = os.getenv('DB_HOST')
      db_port = int(os.getenv('DB_PORT'))
      db_name = os.getenv('POSTGRES_DB')
      db_user = os.getenv('POSTGRES_USER')
      db_password = os.getenv('POSTGRES_PASSWORD')
      logger.debug('Connecting to database: %s@%s:%s/%s', db_user, db_host, db_port, db_name)

      self.connection = psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
      )
      self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
      logger.info('Successfully connected to database')
      return True
    except Exception as e:
      logger.error('Failed to connect to database: %s', e)
      return False

  def disconnect_db(self):
    if self.cursor:
      self.cursor.close()
    if self.connection:
      self.connection.close()
    logger.info('Disconnected from database')

  def setup_database(self) -> bool:
    try:
      self.cursor.execute('CREATE EXTENSION IF NOT EXISTS vector;')
      self.cursor.execute('DROP TABLE IF EXISTS codebase_chunks;')
      create_table_query = f"""
            CREATE TABLE codebase_chunks (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_extension TEXT,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                chunk_content TEXT NOT NULL,
                chunk_hash VARCHAR(64) NOT NULL UNIQUE,
                file_size INTEGER,
                file_modified TIMESTAMP,
                language TEXT,
                embedding halfvec({self.embedding_dim}),
                token_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX codebase_chunks_embedding_idx
            ON codebase_chunks USING hnsw (embedding halfvec_cosine_ops);

            CREATE INDEX codebase_chunks_file_path_idx ON codebase_chunks(file_path);
            CREATE INDEX codebase_chunks_language_idx ON codebase_chunks(language);
            CREATE INDEX codebase_chunks_file_extension_idx ON codebase_chunks(file_extension);
            CREATE INDEX codebase_chunks_hash_idx ON codebase_chunks(chunk_hash);
            """
      self.cursor.execute(create_table_query)
      self.connection.commit()
      logger.info('Database schema setup completed successfully')
      return True
    except Exception as e:
      logger.error('Failed to setup database: %s', e)
      return False

  def _chunk_content(self, content: str) -> List[str]:
    """Split content into overlapping chunks suitable for embedding."""
    # Tokenize the content
    tokens = self.tokenizer.encode(content)
    if len(tokens) <= self.max_tokens_per_chunk:
      return [content]

    chunks = []
    start = 0

    while start < len(tokens):
      end = min(start + self.max_tokens_per_chunk, len(tokens))
      chunk_tokens = tokens[start:end]
      chunk_content = self.tokenizer.decode(chunk_tokens)
      chunks.append(chunk_content)

      # Move start forward, accounting for overlap
      if end == len(tokens):
        break
      start = end - self.chunk_overlap

    return chunks

  def _embed_text(self, text: str) -> list | None:
    try:
      embedding = self.embedding_model.encode(text, normalize_embeddings=True)
      return embedding.tolist()
    except Exception as e:
      logger.error('Failed to generate embedding: %s', e)
      return None

  def _insert_chunk(self, file_path: Path, chunk_index: int, chunk_content: str,
                    file_stats: os.stat_result) -> bool:
    """Insert a single chunk into the database."""
    try:
      # Prepare metadata
      rel_path = str(file_path.relative_to(self.project_root))
      file_name = file_path.name
      file_extension = file_path.suffix
      language = _detect_language(file_path)
      file_modified = datetime.fromtimestamp(file_stats.st_mtime)
      token_count = len(self.tokenizer.encode(chunk_content))
      chunk_hash = _compute_chunk_hash(rel_path, chunk_index, chunk_content)

      # Check if a chunk already exists with the same hash
      self.cursor.execute('SELECT id FROM codebase_chunks WHERE chunk_hash = %s', (chunk_hash,))
      if self.cursor.fetchone():
        logger.debug('Chunk already exists: %s:%s', rel_path, chunk_index)
        return True

      # Generate embedding
      embedding = self._embed_text(chunk_content)
      if embedding is None:
        return False

      # Convert embedding to string format for pgvector
      embedding_str = f'[{",".join(map(str, embedding))}]'

      # Insert chunk
      insert_query = """
                     INSERT INTO codebase_chunks (file_path, file_name, file_extension, chunk_index, chunk_content,
                                                  chunk_hash, file_size, file_modified, language, embedding,
                                                  token_count)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                     """

      self.cursor.execute(insert_query, (
        rel_path, file_name, file_extension, chunk_index, chunk_content,
        chunk_hash, file_stats.st_size, file_modified, language, embedding_str, token_count
      ))

      self.connection.commit()
      return True

    except Exception as e:
      logger.error('Failed to insert chunk %s:%s: %s', file_path, chunk_index, e)
      self.connection.rollback()
      return False

  def index_file(self, file_path: Path) -> bool:
    try:
      file_stats = file_path.stat()

      with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
      chunks = self._chunk_content(content)

      # Insert each chunk
      success_count = 0
      for chunk_index, chunk_content in enumerate(chunks):
        if self._insert_chunk(file_path, chunk_index, chunk_content, file_stats):
          success_count += 1
          self.stats['chunks_created'] += 1

      if success_count == len(chunks):
        self.stats['files_processed'] += 1
        logger.info('Indexed: %s (%s chunks)', file_path.relative_to(self.project_root), len(chunks))
        return True
      else:
        logger.warning('Partial success: %s (%s/%s chunks)', file_path.relative_to(self.project_root),
                       success_count, len(chunks))
        return False

    except Exception as e:
      logger.error('Failed to index file %s: %s', file_path, e)
      self.stats['errors'] += 1
      return False

  def discover_files(self) -> Generator[Path, None, None]:
    tracked_files = self._get_git_tracked_files()
    for file_path in tracked_files:
      if _should_index_file(file_path):
        yield file_path
      else:
        self.stats['files_skipped'] += 1

  def index_codebase(self) -> bool:
    logger.info('Starting codebase indexing...')

    start_time = datetime.now()
    try:
      for file_path in self.discover_files():
        self.index_file(file_path)
        # Progress logging every 50 files
        if self.stats['files_processed'] % 50 == 0:
          logger.info('Progress: %s files processed, %s chunks created', self.stats['files_processed'],
                      self.stats['chunks_created'])

    except KeyboardInterrupt:
      logger.warning('Indexing interrupted by user')
      return False
    except Exception as e:
      logger.error('Indexing failed: %s', e)
      return False

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info('Indexing completed!')
    logger.info('Statistics:')
    logger.info('   Files processed: %s', self.stats['files_processed'])
    logger.info('   Chunks created: %s', self.stats['chunks_created'])
    logger.info('   Files skipped: %s', self.stats['files_skipped'])
    logger.info('   Errors: %s', self.stats['errors'])
    logger.info('   Duration: %s', duration)

    return True

  def search_code(self, query: str, limit: int = 10) -> List[Dict]:
    try:
      query_embedding = self._embed_text(query)
      if query_embedding is None:
        return []

      search_query = """
                     SELECT file_path,
                            file_name,
                            language,
                            chunk_index,
                            chunk_content,
                            embedding <=>
                            %s::vector       as distance,
                            1 - (embedding <=>
                                 %s::vector) as similarity
                     FROM codebase_chunks
                     WHERE embedding IS NOT NULL
                     ORDER BY embedding <=> %s::vector
                     LIMIT %s;
                     """
      query_vector = f'[{",".join(map(str, query_embedding))}]'
      self.cursor.execute(search_query, (query_vector, query_vector, query_vector, limit))
      results = self.cursor.fetchall()

      return [dict(row) for row in results]

    except Exception as e:
      logger.error('Search failed: %s', e)
      return []


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Codebase Indexer',
    epilog='Example: python codebase_indexer.py --project=/path/to/project'
  )
  parser.add_argument(
    '--project',
    type=Path,
    default=Path(os.getcwd()),
    help='Project root directory',
  )
  args = parser.parse_args()

  logger.info('codebase_indexer - Project root: %s', args.project)
  indexer = CodebaseIndexer(args.project)
  try:
    if not indexer.connect_db():
      sys.exit(1)
    if not indexer.setup_database():
      sys.exit(1)
    if not indexer.index_codebase():
      sys.exit(1)

    # Test search
    logger.info('Testing search functionality...')
    test_queries = [
      'Angular component with search functionality',
      'HTTP service for API calls',
      'Authentication and login',
      'Database models and entities'
    ]
    for query in test_queries:
      logger.info("\n--- Search: '%s' ---", query)
      results = indexer.search_code(query, limit=3)
      if results:
        for i, result in enumerate(results, 1):
          logger.info('%s. %s (%s) - similarity: %.3f', i, result['file_path'], result['language'],
                      result['similarity'])
          preview = result['chunk_content'][:100].replace('\n', ' ')
          logger.info('   Preview: %s...', preview)
      else:
        logger.info('   No results found')

    logger.info('Indexing and testing completed successfully!')

  except KeyboardInterrupt:
    logger.info('Operation cancelled by user')
  except Exception as e:
    logger.error('Unexpected error: %s', e)
    import traceback

    traceback.print_exc()
  finally:
    indexer.disconnect_db()
