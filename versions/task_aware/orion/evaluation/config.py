from __future__ import annotations



import os

from dataclasses import dataclass

from pathlib import Path



from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI

from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.llms import LangchainLLMWrapper



from orion import env_utils  # noqa: F401  # loads .env via python-dotenv





def _default_embedding_model() -> str:

    """Resolve the default local embedding model relative to this module.



    Walks upward from ``orion/evaluation`` so the default works whether the

    code is executed from a normal checkout or from a git worktree.

    """

    here = Path(__file__).resolve().parent

    for parent in [here, *here.parents]:

        candidate = parent / "models" / "BAAI" / "bge-m3"

        if candidate.exists():

            return str(candidate)

    # 常规布局为 <root>/orion/evaluation；无需假设更多父目录存在。

    return str(here.parents[1] / "models" / "BAAI" / "bge-m3")





DEFAULT_EMBEDDING_MODEL = _default_embedding_model()





@dataclass

class EvalConfig:

    """评估框架配置。"""



    base_url: str

    api_key: str

    model: str

    metrics_threshold: float = 0.5

    auto_approve: bool = True

    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL



    @classmethod

    def from_env(cls) -> "EvalConfig":

        # 显式设为空字符串表示“禁用 embedding”，未设置时才回退到默认本地模型。

        raw_embed = os.environ.get("RAGAS_EMBEDDING_MODEL", None)

        if raw_embed is None:

            embed_name = DEFAULT_EMBEDDING_MODEL

        else:

            embed_name = raw_embed

        return cls(

            base_url=os.getenv("RAGAS_BASE_URL", "").rstrip("/"),

            api_key=os.getenv("RAGAS_API_KEY", ""),

            model=os.getenv("RAGAS_MODEL", "gpt-5.6-terra"),

            metrics_threshold=float(os.getenv("RAGAS_THRESHOLD", "0.5")),

            auto_approve=os.getenv("ORION_AUTO_APPROVE", "1") == "1",

            embedding_model_name=embed_name,

        )



    def get_judge_llm(self):

        """返回 RAGAS 可用的 LLM wrapper。"""

        if not self.base_url or not self.api_key:

            raise ValueError("RAGAS_BASE_URL and RAGAS_API_KEY must be set")

        chat = ChatOpenAI(

            model=self.model,

            api_key=self.api_key,

            base_url=self.base_url,

            temperature=0.0,

            timeout=120,

            max_retries=5,

        )

        return LangchainLLMWrapper(chat)



    def get_judge_embeddings(self):

        """返回 RAGAS 可用的 embedding wrapper；本地模型无法加载时返回 None。



        强制 CPU：kg_rag 检索的 bge-m3 已占满 GPU，评审 embeddings 再上 GPU 会 OOM。

        若路径不存在或加载失败（内存不足等），跳过依赖 embeddings 的指标

        （answer_relevancy），避免阻塞其余 LLM-based 指标。

        """

        if not os.path.exists(self.embedding_model_name):

            print(

                f"[EvalConfig] embedding model path not found: {self.embedding_model_name}; "

                "answer_relevancy will be skipped",

                flush=True,

            )

            return None

        try:

            # Windows 上 torch_cpu.dll 在并行初始化 bge-m3 时可能访问冲突；

            # 评测 embedding 不走 GPU，固定单线程以保证评测可重复完成。

            import torch

            try:

                torch.set_num_threads(1)

                torch.set_num_interop_threads(1)

            except RuntimeError:

                # 同进程测试/调用可能已经初始化线程池；此时不能重设，

                # 但也不应把可用 embedding 误判为加载失败。

                pass

            embeddings = HuggingFaceEmbeddings(

                model_name=self.embedding_model_name,

                model_kwargs={"device": "cpu"},

            )

            return LangchainEmbeddingsWrapper(embeddings)

        except Exception as exc:  # noqa: BLE001

            print(

                f"[EvalConfig] failed to load embedding model ({exc}); "

                "answer_relevancy will be skipped",

                flush=True,

            )

            return None





DEFAULT_METRICS = [

    "context_precision",

    "context_recall",

    "faithfulness",

    "answer_relevancy",

]

