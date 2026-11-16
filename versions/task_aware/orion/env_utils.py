"""Read only an explicitly supplied environment file; never search parent folders."""
import os
from pathlib import Path
from dotenv import load_dotenv
_file=os.environ.get('ORION_ENV_FILE','').strip()
if _file:load_dotenv(Path(_file),override=False)
DEEPSEEK_API_KEY=os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL=os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com')
EMBEDDING_MODEL=os.getenv('EMBEDDING_MODEL','BAAI/bge-m3')
OPENROUTER_API_KEY=os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL=os.getenv('OPENROUTER_BASE_URL')
MLFZ_API_KEY=os.getenv('MLFZ_API_KEY')
MLFZ_BASE_URL=os.getenv('MLFZ_BASE_URL')
