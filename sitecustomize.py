# Enable both historical import spellings, independent of the caller's directory.
import os,sys
from pathlib import Path
r=Path(__file__).resolve().parent
sys.path[:0]=[str(r/'src'),str(r/'src/orion')]
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
