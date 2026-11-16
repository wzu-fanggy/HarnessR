import os
import warnings
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from env_utils import DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY,EMBEDDING_MODEL,MLFZ_BASE_URL,MLFZ_API_KEY
from model_endpoints import normalize_openai_base_url, resolve_grounding_endpoint_config
from model_config import AGENT_MAX_OUTPUT_TOKENS, AGENT_MODEL, AGENT_TEMPERATURE

# 可选的 LLM 响应缓存：重复 prompt（同模型参数）命中缓存，显著降低评估/E2E 复测成本。
# 默认关闭；开启方式：ORION_LLM_CACHE=1，或 ORION_LLM_CACHE=/path/to/cache.db
_orion_llm_cache = os.environ.get("ORION_LLM_CACHE", "")
if _orion_llm_cache and _orion_llm_cache.lower() not in ("0", "false", "no"):
    if _orion_llm_cache.lower() in ("1", "true", "yes"):
        _cache_path = Path(__file__).resolve().parent / "data" / "llm_cache.db"
    else:
        _cache_path = Path(_orion_llm_cache)
    _cache_path.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(_cache_path)))
    print(f"[llms] LLM response cache enabled: {_cache_path}", flush=True)

def _resolve_agent_model_overrides():
    """解析实验主模型 env 覆盖。

    未设置 ORION_AGENT_* 时返回与历史硬编码完全一致的默认值：
    provider=deepseek、模型名取 model_config.AGENT_MODEL、端点沿用 DEEPSEEK_BASE_URL 链，
    并携带 DeepSeek 专有的思考开关 extra_body。ORION_AGENT_BASE_URL 仅在显式设置时
    覆盖端点；非 deepseek provider 且未显式给端点时回退 RAGAS_BASE_URL
    （与 grounding judge 的 ORION_GROUNDING_* > RAGAS_* 惯例同源）。
    extra_body 只在 provider 为 deepseek（大小写不敏感）时下发，
    其他 OpenAI 兼容端点不识别该字段。ORION_AGENT_TEMPERATURE 未设置时
    保持 model_config.AGENT_TEMPERATURE；显式设置时必须是有限浮点数。
    ORION_AGENT_TOP_P 由独立 helper 解析，默认显式固定为 1.0。
    """
    provider = (os.environ.get("ORION_AGENT_PROVIDER") or "deepseek").strip()
    model = (os.environ.get("ORION_AGENT_MODEL") or AGENT_MODEL).strip()
    base_url_override = (os.environ.get("ORION_AGENT_BASE_URL") or "").strip()
    temperature_override = (os.environ.get("ORION_AGENT_TEMPERATURE") or "").strip()
    if base_url_override:
        base_url = normalize_openai_base_url(base_url_override, label="ORION_AGENT_BASE_URL")
    elif provider.lower() == "deepseek":
        base_url = normalize_openai_base_url(DEEPSEEK_BASE_URL, label="DEEPSEEK_BASE_URL")
    else:
        base_url = normalize_openai_base_url(
            (os.environ.get("RAGAS_BASE_URL") or "").strip(), label="RAGAS_BASE_URL"
        )
    thinking = (os.environ.get("ORION_AGENT_THINKING") or "disabled").strip().lower()
    if thinking not in {"enabled", "disabled"}:
        raise ValueError("ORION_AGENT_THINKING must be enabled or disabled")
    extra_body = {"thinking": {"type": thinking}} if provider.lower() == "deepseek" else None
    temperature = AGENT_TEMPERATURE if not temperature_override else float(temperature_override)
    if not float("-inf") < temperature < float("inf"):
        raise ValueError("ORION_AGENT_TEMPERATURE must be finite")
    return provider, model, base_url, extra_body, temperature


def _resolve_agent_api_key(provider: str) -> str:
    """解析主模型 API key；默认与历史行为逐字节一致（DeepSeek key）。

    优先级：ORION_AGENT_API_KEY > provider 默认链（deepseek → DEEPSEEK_API_KEY；
    其他 provider → RAGAS_API_KEY，与 grounding judge 惯例同源）。
    """

    explicit = (os.environ.get("ORION_AGENT_API_KEY") or "").strip()
    if explicit:
        return explicit
    if provider.lower() == "deepseek":
        return DEEPSEEK_API_KEY
    return (os.environ.get("RAGAS_API_KEY") or "").strip()


def _resolve_agent_top_p() -> float:
    """解析主模型 top_p；默认显式固定 1.0，实验可通过 env hash-bind。"""

    raw = (os.environ.get("ORION_AGENT_TOP_P") or "1.0").strip()
    top_p = float(raw)
    if not float("-inf") < top_p < float("inf"):
        raise ValueError("ORION_AGENT_TOP_P must be finite")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("ORION_AGENT_TOP_P must be in (0, 1]")
    return top_p


# P8e：主模型 env 覆盖在模块加载时解析一次；不设 ORION_AGENT_* 时行为与历史版本零差异。
_agent_provider, _agent_model, _main_base_url, _agent_extra_body, _agent_temperature = _resolve_agent_model_overrides()
_agent_top_p = _resolve_agent_top_p()
_agent_api_key = _resolve_agent_api_key(_agent_provider)

def _build_main_chat_model(*, request_timeout: int):
    """构造主模型客户端。

    provider 为 deepseek 时改用保留推理内容的子类：思考模式下 DeepSeek 要求把
    上一轮 assistant 的 ``reasoning_content`` 原样发回，而出站转换
    （``langchain_openai`` 的 ``_convert_message_to_dict``）只还原 ``tool_calls``
    与 ``function_call``，会把该字段丢掉，使多轮工具循环的第二轮请求直接返回
    400。其余 provider 保持原生 ``init_chat_model`` 行为。
    """

    common = dict(
        model=_agent_model,
        base_url=_main_base_url,
        api_key=_agent_api_key,
        max_tokens=AGENT_MAX_OUTPUT_TOKENS,
        # 低温（贪心）：工具调用密集的科学设计任务要求行为可复现、结果可审计；
        # temperature=1.0 的 run 间行为方差曾使评估指标随机波动 ±0.3（v17 vs v18），
        # 且同任务可能漏调 skill 声明的必须工具。可溯源/可验证优先于措辞多样性。
        temperature=_agent_temperature,
        top_p=_agent_top_p,
        # 读超时按场景给定：流式 180s（相邻 chunk 间隔超时即视为响应停滞，抛
        # APITimeoutError 而不是无限挂起；正常流式生成 chunk 持续到达，不会误触发）；
        # 非流式 900s（服务端缓冲完整结果后才返回，首字节等待 ≈ 整个生成时长，
        # 避免长报告生成被误判为停滞）。
        request_timeout=request_timeout,
        # SDK 隐式重试关闭；连接、超时与不完整响应统一由 main 的显式有界策略处理并计数。
        max_retries=0,
        extra_body=_agent_extra_body,
    )
    if _agent_provider.lower() == "deepseek":
        from core.deepseek_reasoning_roundtrip import ReasoningPreservingChatDeepSeek

        return ReasoningPreservingChatDeepSeek(**common)
    return init_chat_model(**common, model_provider=_agent_provider)


main_model = _build_main_chat_model(request_timeout=180)

# 非流式变体：供 E2E 测试/脚本等无可视化场景使用（ORION_LLM_STREAM=0）。
main_model_nonstream = _build_main_chat_model(request_timeout=900)

# --- 接地校验独立模型 -------------------------------------------------------
# 与生成模型不同源，避免"自己审自己"的自检盲点（v22 归因：过度概括+参数知识
# 合计占剩余 MISS 的 ~45%，同模型校验对这两类判不出来）。懒加载单例；
# 未配置时返回 None，由调用方回退主模型（降级路径，行为同 v22）。
_grounding_model = None
_grounding_model_resolved = False


def get_grounding_model():
    """返回接地校验用的独立模型；未配置时返回 None。

    配置优先级：ORION_GROUNDING_* > RAGAS_*（与评估 judge 同源，1314mc 中转）。
    """
    global _grounding_model, _grounding_model_resolved
    if _grounding_model_resolved:
        return _grounding_model
    config = resolve_grounding_endpoint_config()
    if config is None:
        _grounding_model = None
        _grounding_model_resolved = True
        return None
    if config.insecure_http:
        warnings.warn(
            "Independent grounding judge is using explicitly allowed plaintext HTTP; "
            "the API credential is not protected by TLS",
            RuntimeWarning,
            stacklevel=2,
        )
    _grounding_model = init_chat_model(
        model=config.model,
        model_provider="openai",
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=0.0,
        max_tokens=8192,
        # 非流式：校验 prompt 携带全部证据 contexts，响应较慢，超时放宽。
        # v60：默认 120s，可通过环境变量覆盖；避免单条校验挂死 5 分钟拖垮评估。
        request_timeout=config.request_timeout,
        max_retries=config.max_retries,
    )
    _grounding_model_resolved = True
    return _grounding_model


from langchain_core.embeddings import Embeddings
import requests

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model, api_key, base_url):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip('/') + '/embeddings'

    def embed_documents(self, texts):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": texts,"dimension":1024}
        response = requests.post(self.base_url, headers=headers, json=payload)
        body = response.json()
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            preview = response.text[:500]
            raise RuntimeError(
                f"Embedding API returned unexpected response: status={response.status_code}, body={preview}"
            )
        return [item['embedding'] for item in data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# 使用示例
embedding_model = OpenRouterEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=MLFZ_API_KEY,
    base_url=MLFZ_BASE_URL
)
# a=embedding_model.embed_documents(['hello world'])
# print(a)
# vector = embedding_model.embed_query('你好')
# print(vector)
# res =main_model.invoke('你好')
# print(res)
