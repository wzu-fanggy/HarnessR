"""Workbench API 包初始化：确保父仓库包可在独立 API 工作目录中解析。"""
from __future__ import annotations

import os
import sys
from pathlib import Path


_DEFAULT_KG_CREATE_ROOT = Path(__file__).resolve().parents[4]
_KG_CREATE_ROOT = Path(os.environ.get("KG_CREATE_ROOT", _DEFAULT_KG_CREATE_ROOT))
if str(_KG_CREATE_ROOT) not in sys.path:
    sys.path.insert(0, str(_KG_CREATE_ROOT))
