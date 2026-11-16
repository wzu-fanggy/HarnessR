import os

from dotenv import load_dotenv
load_dotenv()

# Milvus
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "multimodal_rag"

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

# 检索默认参数
DEFAULT_TOP_K = 5
