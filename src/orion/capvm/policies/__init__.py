"""Capability Virtual Memory 的 residency replacement policies。"""

from .lru_cost import LRUCostEvictionPolicy
from .laya_reuse import LayaDecisionModel, LayaReuseEvictionPolicy

__all__ = ["LRUCostEvictionPolicy", "LayaDecisionModel", "LayaReuseEvictionPolicy"]
