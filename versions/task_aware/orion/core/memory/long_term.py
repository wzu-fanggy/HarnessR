from __future__ import annotations



import asyncio

import os

import sqlite3

from pathlib import Path

from typing import Any



from langgraph.store.base import BaseStore, IndexConfig

from langgraph.store.memory import InMemoryStore



try:

    from langgraph.store.sqlite import SqliteStore

except Exception:  # pragma: no cover - older langgraph fallback

    SqliteStore = None  # type: ignore[misc, assignment]



from orion.tools.rag.service_client import get_root_kg_service_client





class _DummyEmbedder:

    """在内存/页面文件不足时的降级 embedder，返回零向量以保活长期记忆接口。"""



    dims: int = 1024



    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        return [[0.0] * self.dims for _ in texts]



    def embed_query(self, text: str) -> list[float]:

        return [0.0] * self.dims





def _default_db_path() -> str:

    """默认持久化向量数据库路径，与 FactsStore 分文件避免冲突。"""

    root = Path(__file__).resolve().parents[2]

    db = root / "data" / "long_term_memory_vectors.db"

    db.parent.mkdir(parents=True, exist_ok=True)

    return str(db)





class _ServiceEmbedder:

    """由根服务提供 bge-m3 embedding 的轻量适配器。"""



    def __init__(self) -> None:

        self.dims = 1024



    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        result = get_root_kg_service_client().call("/v1/embed", {"texts": texts})

        vectors = result.get("vectors", []) if isinstance(result, dict) else []

        if not isinstance(vectors, list) or len(vectors) != len(texts):

            raise RuntimeError("root KG service returned invalid embedding vectors")

        self.dims = int(result.get("dimension", self.dims))

        return vectors



    def embed_query(self, text: str) -> list[float]:

        return self.embed_documents([text])[0]





class LongTermMemory:

    """基于根服务 bge-m3 embedding + LangGraph Store 的长期记忆。



    默认使用 ``SqliteStore`` 做持久化，重启后记忆不丢失；

    若 SQLite 不可用或初始化失败，自动降级为 ``InMemoryStore``。



    Embedder 延迟创建客户端：只在第一次 remember/recall 时访问根服务，避免

    API/CLI 进程直接加载大模型。根服务不可用时自动降级为零向量 embedder，

    保证接口可用但语义搜索失效。

    """



    _shared_embedder: _ServiceEmbedder | _DummyEmbedder | None = None



    def __init__(

        self,

        store: BaseStore | None = None,

        model_path: str | None = None,

        db_path: str | None = None,

    ):

        # 保留 model_path 参数的调用兼容性；模型路径由根服务管理。

        del model_path

        self._db_path = db_path or os.environ.get("LTM_VECTORS_DB_PATH") or _default_db_path()

        self._store = store



    @property

    def embedder(self) -> _ServiceEmbedder:

        # 根服务恢复后应自动重试，不能把一次临时错误永久缓存为零向量。

        if not isinstance(LongTermMemory._shared_embedder, _ServiceEmbedder):

            LongTermMemory._shared_embedder = _ServiceEmbedder()

        return LongTermMemory._shared_embedder



    @property

    def store(self) -> BaseStore:

        if self._store is None:

            self._store = self._create_store()

        return self._store



    def _create_store(self) -> BaseStore:

        dims = getattr(self.embedder, "dims", _DummyEmbedder.dims)

        index = IndexConfig(embed=self._embed, dims=int(dims))

        if SqliteStore is not None:

            try:

                conn = sqlite3.connect(self._db_path, check_same_thread=False, isolation_level=None)

                return SqliteStore(conn=conn, index=index)

            except Exception as exc:  # noqa: BLE001

                print(

                    f"[LongTermMemory] failed to open SQLite store at {self._db_path} ({exc}); "

                    "falling back to InMemoryStore",

                    flush=True,

                )

        return InMemoryStore(index=index)



    def _embed(self, texts: list[str]) -> list[list[float]]:

        try:

            return self.embedder.embed_documents(texts)

        except Exception as exc:  # noqa: BLE001

            print(

                f"[LongTermMemory] root embedding service unavailable ({exc}); "

                "falling back to dummy zero embeddings",

                flush=True,

            )

            LongTermMemory._shared_embedder = _DummyEmbedder()

            return LongTermMemory._shared_embedder.embed_documents(texts)



    async def remember(

        self,

        namespace: tuple[str, ...],

        key: str,

        content: dict[str, Any],

    ) -> None:

        """把结构化内容写入长期记忆。优先使用异步 aput；若底层 store 未实现（如 SqliteStore），

        则回退到同步 put 并在线程中执行。"""

        store = self.store

        try:

            await store.aput(namespace, key, content)

        except NotImplementedError:

            await asyncio.to_thread(store.put, namespace, key, content)



    async def recall(

        self,

        namespace: tuple[str, ...],

        query: str,

        top_k: int = 5,

    ) -> list[dict[str, Any]]:

        """按语义搜索长期记忆。优先使用异步 asearch；若底层 store 未实现，

        则回退到同步 search 并在线程中执行。"""

        store = self.store

        try:

            items = await store.asearch(namespace, query=query, limit=top_k)

        except NotImplementedError:

            items = await asyncio.to_thread(store.search, namespace, query=query, limit=top_k)

        return [

            {

                "namespace": item.namespace,

                "key": item.key,

                "value": item.value,

                "score": getattr(item, "score", None),

            }

            for item in items

        ]

