import os
"""Default configuration for kg_rag."""
from release_paths import release_path
from pathlib import Path

NEO4J_URI = os.environ.get('NEO4J_URI','bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER','neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD','')
NEO4J_DATABASE = os.environ.get('NEO4J_DATABASE','kg-v1')

EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL','BAAI/bge-m3')
PARENT_MAX_LEN = 3000
CHILD_MAX_LEN = 1500
CHILD_OVERLAP = 200
PAPER_BATCH_SIZE = 50
CHUNK_INSERT_BATCH = 64
REL_INSERT_BATCH = 200

CHECKPOINT_FILE = Path("vector_kb_checkpoint.json")

DEFAULT_MINERU_ROOTS = [
    Path(release_path('${PROJECT_SOURCE}\\mineru_output')),
    Path(release_path('${PROJECT_SOURCE}\\mineru_output_arxiv_400')),
]
ARXIV_METADATA = Path(release_path('${PROJECT_SOURCE}\\papers_arxiv_400\\filtered_metadata.json'))
VECTOR_INDEX_NAME = "chunk_embedding"
