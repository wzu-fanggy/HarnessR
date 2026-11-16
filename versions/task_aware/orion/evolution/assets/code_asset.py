from __future__ import annotations

from typing import Literal

from orion.evolution.assets.base import ASSET_CLASS_REGISTRY, VersionedAsset
from orion.evolution.eval.base import EvalResult


class CodeAsset(VersionedAsset):
    """基于代码文本的版本化资产占位类。"""

    asset_type: Literal["code"] = "code"

    def _apply_patch_to_content(self, content: str | dict, patch_diff: str) -> str | dict:
        if not content:
            return patch_diff
        return f"{content}\n{patch_diff}"

    async def validate(self) -> EvalResult:
        """默认验证：占位，始终通过。"""
        return EvalResult(score=1.0, reasoning="code asset validation stub", passed=True)


ASSET_CLASS_REGISTRY["code"] = CodeAsset
