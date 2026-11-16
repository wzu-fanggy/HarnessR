from __future__ import annotations

import json
from typing import Literal

from orion.evolution.assets.base import ASSET_CLASS_REGISTRY, VersionedAsset
from orion.evolution.assets.storage import FileSystemAssetStorage
from orion.evolution.eval.base import EvalResult


class ConfigAsset(VersionedAsset):
    """基于 JSON 配置的版本化资产。"""

    asset_type: Literal["config"] = "config"

    def __init__(self, asset_id: str, content: dict | None = None, **kwargs):
        kwargs.setdefault("asset_type", "config")
        kwargs.setdefault("versions", {})
        kwargs.setdefault("active_version", "")
        kwargs.setdefault("eval_history", [])
        super().__init__(asset_id=asset_id, **kwargs)
        if content:
            self.apply_patch(json.dumps(content), metadata={"source": "initial"})

    def _apply_patch_to_content(self, content: str | dict, patch_diff: str) -> str | dict:
        current = dict(content) if isinstance(content, dict) else {}
        patch = json.loads(patch_diff)
        if isinstance(patch, dict):
            current.update(patch)
            return current
        return patch

    async def validate(self) -> EvalResult:
        """默认验证：检查当前配置是否为合法 JSON 对象。"""
        content = self.active_content
        if isinstance(content, dict):
            return EvalResult(score=1.0, reasoning="valid config dict", passed=True)
        return EvalResult(score=0.0, reasoning="config content is not a dict", passed=False)


def load_config_asset(asset_id: str, storage: FileSystemAssetStorage) -> ConfigAsset | None:
    return storage.load(asset_id, "config", ConfigAsset)


ASSET_CLASS_REGISTRY["config"] = ConfigAsset
