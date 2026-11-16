"""可由 Agent runtime 显式获取/释放的根 KG 服务资源。"""

from __future__ import annotations

import threading
from time import monotonic
from typing import Any

from neo4j import GraphDatabase

from kg_rag.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


class ReloadableNeo4jDriver:
    """保持稳定 Python 对象身份、但可关闭并重新建立底层 Neo4j driver。

    kg_rag 的 retriever / query service 可以长期持有本对象；Agent runtime 通过
    ensure_open() 与 close() 控制实际连接池生命周期，而无需重建上层 callable。
    """

    def __init__(
        self,
        *,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: Any | None = None
        self._lock = threading.RLock()
        self.open_count = 0
        self.close_count = 0
        self.last_open_seconds = 0.0

    @property
    def is_open(self) -> bool:
        return self._driver is not None

    def ensure_open(self) -> Any:
        """Create and verify the underlying driver once per resident interval."""

        with self._lock:
            if self._driver is not None:
                return self._driver
            started = monotonic()
            driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            try:
                driver.verify_connectivity()
            except Exception:
                driver.close()
                raise
            self._driver = driver
            self.open_count += 1
            self.last_open_seconds = monotonic() - started
            return driver

    def session(self, *args: Any, **kwargs: Any) -> Any:
        return self.ensure_open().session(*args, **kwargs)

    def execute_query(self, *args: Any, **kwargs: Any) -> Any:
        return self.ensure_open().execute_query(*args, **kwargs)

    def verify_connectivity(self) -> None:
        self.ensure_open().verify_connectivity()

    def close(self) -> bool:
        """Close the underlying connection pool while preserving the proxy object."""

        with self._lock:
            driver = self._driver
            if driver is None:
                return False
            self._driver = None
            driver.close()
            self.close_count += 1
            return True

    def runtime_snapshot(self) -> dict[str, object]:
        return {
            "resource_id": "neo4j-session",
            "resident": self.is_open,
            "connection_slots": 1.0 if self.is_open else 0.0,
            "open_count": self.open_count,
            "close_count": self.close_count,
            "last_open_seconds": self.last_open_seconds,
        }

    def __getattr__(self, name: str) -> Any:
        # neo4j-graphrag may access less common Driver methods/properties. Delegate them
        # through the stable proxy so every such access still respects residency.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.ensure_open(), name)
