"""冻结任务证据 benchmark 与确定性离线评估。"""



import sys

import types

from pathlib import Path



# Allow the benchmark package to be imported when scripts are run directly

# from inside the orion worktree without requiring pytest's conftest setup.

_orion_root = Path(__file__).resolve().parent

_project_root = _orion_root.parent

for _path in (_orion_root, _project_root):

    _s = str(_path)

    if _s not in sys.path:

        sys.path.insert(0, _s)

if "orion" not in sys.modules:

    _orion_pkg = types.ModuleType("orion")

    _orion_pkg.__path__ = [str(_orion_root)]

    sys.modules["orion"] = _orion_pkg



from .metrics_v2 import score_task_artifact_v2



__all__ = ["score_task_artifact_v2"]

