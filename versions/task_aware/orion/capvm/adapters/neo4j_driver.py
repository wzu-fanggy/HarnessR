"""实验用 Neo4j Driver 生命周期：保留连接池，逐次创建 Session。"""

from __future__ import annotations

from time import monotonic
from typing import Any

from ..model import ResidencyLevel
from .neo4j_session import Neo4jSessionCapabilityAdapter


class Neo4jDriverCapabilityAdapter(Neo4jSessionCapabilityAdapter):
    """与根 KG/RAG 服务相同的 Driver/Session 所有权边界。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.driver_open_count = 0
        self.session_creation_count = 0

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current < ResidencyLevel.BOUND <= target and not self.is_connected:
            self._open_driver()

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current >= ResidencyLevel.BOUND > target:
            self._close_driver()

    def close(self) -> None:
        self._close_driver()

    def _open_driver(self) -> None:
        from neo4j import GraphDatabase

        started = monotonic()
        driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            connection_timeout=self.connection_timeout_s,
        )
        try:
            driver.verify_connectivity()
        except Exception:
            driver.close()
            raise
        self._driver = driver
        self.driver_open_count += 1
        self.last_startup_latency_s = monotonic() - started

    def run_read(self, query: str, /, **parameters: object) -> list[dict[str, Any]]:
        with self._query_lock:
            if self._driver is None:
                raise RuntimeError("Neo4j driver capability is not resident")
            session_kwargs: dict[str, object] = {}
            if self.database:
                session_kwargs["database"] = self.database
            with self._driver.session(**session_kwargs) as session:
                self.session_creation_count += 1
                rows = [record.data() for record in session.run(query, **parameters)]
            self.query_count += 1
            return rows

    def _close_driver(self) -> None:
        if self._driver is None:
            return
        started = monotonic()
        driver = self._driver
        self._driver = None
        driver.close()
        self.last_shutdown_latency_s = monotonic() - started
