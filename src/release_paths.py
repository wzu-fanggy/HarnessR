"""Portable path resolution for archived experiment entry points."""
from pathlib import Path
import json, os
ROOT=Path(__file__).resolve().parents[1]
MAP=json.loads((ROOT/'provenance/path_map.json').read_text(encoding='utf8'))
PREFIX={"${ORION_SOURCE}":ROOT/'src/orion',"${PROJECT_SOURCE}":ROOT/'src',
 "${TASK_INTENT_ARCHIVE}":ROOT/'experiments/task_aware',
 "${MCP_EXPERIMENT}":ROOT/'experiments/mcp_sessions',
 "${AUTHOR_WORKSPACE}":ROOT/'data/author_workspace',"${USER_HOME}":ROOT/'outputs'}
def release_path(value):
 value=str(value).replace('\\','/')
 if value in MAP:return str(ROOT/MAP[value])
 for key,base in PREFIX.items():
  if value.startswith(key):return str(base/value[len(key):].lstrip('/'))
 return value
def resolve_json(value):
 if isinstance(value,dict):return {release_path(k):resolve_json(v) for k,v in value.items()}
 if isinstance(value,list):return [resolve_json(v) for v in value]
 if isinstance(value,str) and value.startswith('${'):return release_path(value)
 return value
