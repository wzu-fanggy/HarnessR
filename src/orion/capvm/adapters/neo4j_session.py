"""Real Neo4j session adapter for HCVM integration checks.

The adapter treats a logical scientific database service as a SESSION-backed
capability.  It deliberately keeps Neo4j imports lazy so the core CAPVM package
can still be imported in environments that do not install the Neo4j driver.
"""

from __future__ import annotations

from time import monotonic
from threading import Lock
from typing import Any

from ..model import ResidencyLevel, ResourceVector
from .base import BaseCapabilityAdapter


class Neo4jSessionCapabilityAdapter(BaseCapabilityAdapter):
    """Open/close a real Neo4j driver + session behind the HCVM residency API.

    The current adapter accounts one connection slot while resident.  It does
    not claim that Python-side object size equals the server/network memory
    footprint, so host RAM is intentionally left at zero unless a future
    adapter provides a defensible connection-memory measurement.
    """

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        connection_timeout_s: float = 5.0,
    ) -> None:
        if not uri.strip():
            raise ValueError("uri is required")
        if not user.strip():
            raise ValueError("user is required")
        if connection_timeout_s <= 0:
            raise ValueError("connection_timeout_s must be positive")
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.connection_timeout_s = float(connection_timeout_s)
        self._driver: Any | None = None
        self._session: Any | None = None
        self.last_startup_latency_s = 0.0
        self.last_shutdown_latency_s = 0.0
        self.query_count = 0
        self._query_lock = Lock()

    @property
    def is_connected(self) -> bool:
        return self._driver is not None and self._session is not None

    def declared_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND:
            return ResourceVector()
        return ResourceVector(connection_slots=1.0)

    def measured_cost(self, level: ResidencyLevel) -> ResourceVector:
        if level < ResidencyLevel.BOUND or not self.is_connected:
            return ResourceVector()
        return ResourceVector(connection_slots=1.0)

    def promote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current < ResidencyLevel.BOUND <= target and not self.is_connected:
            self._open_session()

    def demote(self, current: ResidencyLevel, target: ResidencyLevel) -> None:
        if current >= ResidencyLevel.BOUND > target:
            self._close_session()

    def run_read(self, query: str, /, **parameters: object) -> list[dict[str, Any]]:
        """Execute a caller-supplied read-only Cypher query on the resident session.

        HCVM does not attempt to parse or authorize Cypher here.  Production code
        should continue to use the project's existing whitelisted query service;
        this method exists for a narrow runtime integration check where the query
        text is frozen in the benchmark.
        """

        with self._query_lock:
            if self._session is None:
                raise RuntimeError("Neo4j session capability is not resident")
            result = self._session.run(query, **parameters)
            rows = [record.data() for record in result]
            self.query_count += 1
            return rows

    def close(self) -> None:
        self._close_session()

    def _open_session(self) -> None:
        started = monotonic()
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "neo4j Python driver is required for Neo4jSessionCapabilityAdapter"
            ) from exc

        driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            connection_timeout=self.connection_timeout_s,
        )
        try:
            driver.verify_connectivity()
            session_kwargs: dict[str, object] = {}
            if self.database:
                session_kwargs["database"] = self.database
            session = driver.session(**session_kwargs)
        except Exception:
            driver.close()
            raise

        self._driver = driver
        self._session = session
        self.last_startup_latency_s = monotonic() - started

    def _close_session(self) -> None:
        if self._driver is None and self._session is None:
            return
        started = monotonic()
        if self._session is not None:
            try:
                self._session.close()
            finally:
                self._session = None
        if self._driver is not None:
            try:
                self._driver.close()
            finally:
                self._driver = None
        self.last_shutdown_latency_s = monotonic() - started
