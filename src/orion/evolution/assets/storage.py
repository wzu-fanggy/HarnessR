from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePath
from typing import Any

from orion.evolution.assets.base import ASSET_CLASS_REGISTRY, AssetVersion, VersionedAsset


def _default_store_root() -> Path:
    """返回默认资产存储根目录，可通过环境变量 ``ORION_EVOLUTION_STORE`` 覆盖。"""
    env_root = os.environ.get("ORION_EVOLUTION_STORE")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parent.parent / "store"


def _sanitize_asset_id(asset_id: str) -> str:
    """将 asset_id 中的路径分隔符统一替换为安全分隔符，并拒绝路径遍历。"""
    parts = PurePath(asset_id).parts
    if any(part in {"..", "."} for part in parts):
        raise ValueError(f"Invalid asset_id: {asset_id}")
    safe = "_".join(parts)
    # 兜底替换所有残留的路径分隔符
    safe = re.sub(r"[\\\\/]+", "_", safe)
    if not safe:
        raise ValueError(f"Invalid asset_id: {asset_id}")
    return safe


def save_asset(asset: VersionedAsset) -> None:
    """使用默认文件系统存储保存资产。"""
    storage = FileSystemAssetStorage(_default_store_root())
    storage.save(asset)


def load_asset(asset_id: str) -> VersionedAsset | None:
    """使用默认文件系统存储按 asset_id 加载资产，并根据 asset_type 反序列化为对应子类。

    当存储根目录不存在时返回 ``None``，而不是抛出异常。
    """
    root = _default_store_root()
    if not root.exists():
        return None
    safe_id = _sanitize_asset_id(asset_id)
    for asset_type_dir in root.iterdir():
        if not asset_type_dir.is_dir():
            continue
        candidate = asset_type_dir / safe_id
        meta_path = candidate / "asset.json"
        if candidate.is_dir() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            asset_type = meta.get("asset_type")
            asset_class = ASSET_CLASS_REGISTRY.get(asset_type)
            if asset_class is None:
                raise ValueError(f"Unknown asset type: {asset_type}")
            storage = FileSystemAssetStorage(root)
            return storage.load(asset_id, asset_type, asset_class)
    return None


class FileSystemAssetStorage:
    """基于文件系统的版本化资产持久化存储。"""

    def __init__(self, root_dir: Path | str):
        self.root_dir = Path(root_dir)

    def _asset_dir(self, asset_id: str, asset_type: str) -> Path:
        safe_id = _sanitize_asset_id(asset_id)
        return self.root_dir / asset_type / safe_id

    def save(self, asset: VersionedAsset) -> None:
        """将资产元数据与各版本内容写入文件系统。"""
        directory = self._asset_dir(asset.asset_id, asset.asset_type)
        directory.mkdir(parents=True, exist_ok=True)

        asset_meta = {
            "asset_id": asset.asset_id,
            "asset_type": asset.asset_type,
            "active_version": asset.active_version,
            "eval_history": asset.eval_history,
            "version_list": [
                {
                    "version_id": v.version_id,
                    "parent_version": v.parent_version,
                    "metadata": v.metadata,
                    "created_at": v.created_at.isoformat(),
                }
                for v in asset.versions.values()
            ],
        }
        (directory / "asset.json").write_text(
            json.dumps(asset_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        ext = self._ext_for(asset.asset_type)
        for version in asset.versions.values():
            (directory / f"{version.version_id}{ext}").write_text(
                self._serialize_content(version.content, asset.asset_type),
                encoding="utf-8",
            )

        # 删除已不在内存版本列表中的旧版本文件
        expected_files = {directory / f"{version_id}{ext}" for version_id in asset.versions}
        for existing in directory.iterdir():
            if existing.is_dir() or existing.name == "asset.json":
                continue
            if existing not in expected_files:
                existing.unlink()

    def load(self, asset_id: str, asset_type: str, asset_class: type[VersionedAsset]) -> VersionedAsset | None:
        """从文件系统加载资产及其所有版本。"""
        directory = self._asset_dir(asset_id, asset_type)
        if not directory.exists():
            return None
        meta_path = directory / "asset.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        versions: dict[str, AssetVersion] = {}
        for v_meta in meta["version_list"]:
            ext = self._ext_for(asset_type)
            content_path = directory / f"{v_meta['version_id']}{ext}"
            content = self._deserialize_content(
                content_path.read_text(encoding="utf-8"), asset_type
            )
            versions[v_meta["version_id"]] = AssetVersion(
                version_id=v_meta["version_id"],
                parent_version=v_meta.get("parent_version"),
                content=content,
                patch_diff="",  # 不持久化，必要时可重建
                metadata=v_meta.get("metadata", {}),
                created_at=v_meta.get("created_at"),
            )
        return asset_class(
            asset_id=meta["asset_id"],
            asset_type=meta["asset_type"],
            versions=versions,
            active_version=meta["active_version"],
            eval_history=meta.get("eval_history", []),
        )

    def _ext_for(self, asset_type: str) -> str:
        return {
            "prompt": ".md",
            "skill": ".SKILL.md",
            "config": ".json",
            "code": ".py",
        }.get(asset_type, ".txt")

    def _serialize_content(self, content: Any, asset_type: str) -> str:
        if asset_type == "config":
            return json.dumps(content, indent=2, ensure_ascii=False)
        return str(content)

    def _deserialize_content(self, raw: str, asset_type: str) -> Any:
        if asset_type == "config":
            return json.loads(raw)
        return raw
