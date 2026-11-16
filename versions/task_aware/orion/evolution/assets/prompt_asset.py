from __future__ import annotations

import re
from typing import Literal

from orion.evolution.assets.base import ASSET_CLASS_REGISTRY, VersionedAsset
from orion.evolution.assets.storage import FileSystemAssetStorage
from orion.evolution.eval.base import EvalResult


class PromptAsset(VersionedAsset):
    """基于文本提示词的版本化资产。"""

    asset_type: Literal["prompt"] = "prompt"

    def __init__(self, asset_id: str, content: str = "", **kwargs):
        kwargs.setdefault("asset_type", "prompt")
        kwargs.setdefault("versions", {})
        kwargs.setdefault("active_version", "")
        kwargs.setdefault("eval_history", [])
        super().__init__(asset_id=asset_id, **kwargs)
        if content:
            self.apply_patch(content, metadata={"source": "initial"})

    def _apply_patch_to_content(self, content: str | dict, patch_diff: str) -> str | dict:
        if not isinstance(content, str):
            return patch_diff
        match = re.match(r"^s/([^/]*)/([^/]*)/$", patch_diff)
        if match:
            old, new = match.groups()
            return content.replace(old, new, 1)
        return patch_diff

    async def validate(self) -> EvalResult:
        """默认验证：检查 prompt 内容非空。"""
        content = self.active_content
        if isinstance(content, str) and content.strip():
            return EvalResult(score=1.0, reasoning="prompt is non-empty", passed=True)
        return EvalResult(score=0.0, reasoning="prompt is empty", passed=False)


def load_prompt_asset(asset_id: str, storage: FileSystemAssetStorage) -> PromptAsset | None:
    return storage.load(asset_id, "prompt", PromptAsset)


ASSET_CLASS_REGISTRY["prompt"] = PromptAsset
