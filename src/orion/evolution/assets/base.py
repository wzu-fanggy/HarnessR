from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from orion.evolution.eval.base import EvalResult


class AssetVersion(BaseModel):
    """版本化资产的一个具体版本。"""

    version_id: str
    parent_version: str | None = None
    content: str | dict = ""
    patch_diff: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


#: 资产类型字符串到具体 VersionedAsset 子类的映射，供 load_asset 动态反序列化使用。
ASSET_CLASS_REGISTRY: dict[str, type[VersionedAsset]] = {}


class VersionedAsset(BaseModel, ABC):
    """可版本化资产的抽象基类，提供版本链、回滚与评估结果提交能力。"""

    asset_id: str
    asset_type: Literal["prompt", "skill", "config", "code"]
    versions: dict[str, AssetVersion] = Field(default_factory=dict)
    active_version: str = ""
    eval_history: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def active_content(self) -> str | dict:
        """返回当前激活版本的内容；无激活版本时返回空字符串。"""
        if not self.active_version:
            return ""
        return self.versions[self.active_version].content

    def _next_version_id(self) -> str:
        """生成下一个版本 ID，基于已有版本号的最大值，避免修剪后编号冲突。"""
        if not self.versions:
            return "v1"
        max_num = 0
        for version_id in self.versions:
            if version_id.startswith("v"):
                try:
                    max_num = max(max_num, int(version_id[1:]))
                except ValueError:
                    continue
        return f"v{max_num + 1}"

    def apply_patch(self, patch_diff: str, metadata: dict[str, Any] | None = None) -> AssetVersion:
        """基于当前激活版本应用 patch，生成并激活新版本。"""
        parent_id = self.active_version or None
        new_id = self._next_version_id()
        # 初始版本以自身内容作为起点；后续版本在当前内容上应用 patch
        current = self.active_content if parent_id else ""
        new_content = self._apply_patch_to_content(current, patch_diff)
        version = AssetVersion(
            version_id=new_id,
            parent_version=parent_id,
            content=new_content,
            patch_diff=patch_diff,
            metadata=metadata or {},
        )
        self.versions[new_id] = version
        self.active_version = new_id
        return version

    def _version_sort_key(self, version: AssetVersion) -> tuple:
        """返回版本的排序键：先按创建时间，再按版本号数值，避免字符串排序 v10 < v2。"""
        num = 0
        if version.version_id.startswith("v"):
            try:
                num = int(version.version_id[1:])
            except ValueError:
                pass
        created_at = version.created_at or datetime.min.replace(tzinfo=timezone.utc)
        return (created_at, num)

    def trim_versions(self, keep: int = 10) -> None:
        """仅保留最近的 keep 个版本；active_version 永远不会被删除。"""
        if len(self.versions) <= keep:
            return
        sorted_versions = sorted(
            self.versions.values(),
            key=self._version_sort_key,
        )
        to_remove = len(sorted_versions) - keep
        removed = 0
        for version in sorted_versions:
            if removed >= to_remove:
                break
            if version.version_id == self.active_version:
                continue
            del self.versions[version.version_id]
            removed += 1

    @abstractmethod
    def _apply_patch_to_content(self, content: str | dict, patch_diff: str) -> str | dict:
        """子类实现具体的 patch 应用逻辑。"""
        ...

    @abstractmethod
    async def validate(self) -> EvalResult:
        """验证当前激活版本，返回评估结果；子类必须实现。"""
        ...

    def rollback(self, target_version: str) -> None:
        """回滚到指定版本。"""
        if target_version not in self.versions:
            raise ValueError(f"Unknown version: {target_version}")
        self.active_version = target_version

    def commit_if_better(
        self,
        new_version: AssetVersion,
        eval_result: EvalResult,
        min_score: float = 0.6,
        must_improve_parent: bool = True,
    ) -> bool:
        """根据评估结果决定是否保留新版本，否则回滚到父版本。"""
        parent = self.versions.get(new_version.parent_version)
        parent_score = parent.metadata.get("eval_score", 0.0) if parent else 0.0

        if eval_result.score < min_score:
            if parent:
                self.rollback(new_version.parent_version)
            return False
        if must_improve_parent and parent and eval_result.score <= parent_score:
            self.rollback(new_version.parent_version)
            return False

        self.versions[new_version.version_id].metadata["eval_score"] = eval_result.score
        self.eval_history.append({
            "version_id": new_version.version_id,
            "score": eval_result.score,
            "dimensions": eval_result.dimensions,
            "reasoning": eval_result.reasoning,
        })
        return True


class AssetRegistry:
    """内存中的资产注册表，用于运行时的快速查找。"""

    _assets: dict[str, VersionedAsset] = {}

    @classmethod
    def register(cls, asset: VersionedAsset) -> None:
        cls._assets[asset.asset_id] = asset

    @classmethod
    def get(cls, asset_id: str) -> VersionedAsset | None:
        return cls._assets.get(asset_id)

    @classmethod
    def clear(cls) -> None:
        cls._assets.clear()
