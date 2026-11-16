"""Agent tool for fitting electrolyte conductivity temperature dependence."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import ToolException, tool

from .temperature_fit import TemperatureFitInputError, fit_temperature_laws


TRANSPORT_FIT_PROCESS_RESOURCE_ID = "process:transport-fit-kernel"


@tool
def fit_conductivity_temperature_law(request: dict[str, Any]) -> dict[str, Any]:
    """拟合同一电解液配方的实验电导率温度关系。

    request.measurements 至少包含五个不同温度点，每项给出 temperature_c 和
    conductivity_ms_cm。工具同时拟合 Arrhenius 与 VTF，返回 AICc、留一温度误差和
    参数。输入必须是实验测量；结果不证明输运机理，也不授权观测范围外的外推。
    每个温度只能有一个正电导率值；原始重复测量或非正值须先依据明确的数据处理约定整理，
    不能静默删点。输入不合要求时返回工具错误，供修正或升级待决，不会自动清洗数据。
    """

    from orion.core.langgraph_residency import get_active_runtime_resource

    try:
        resource = get_active_runtime_resource(TRANSPORT_FIT_PROCESS_RESOURCE_ID)
        if resource is None:
            return fit_temperature_laws(request)

        from .temperature_fit_process import TemperatureFitProcessAdapter

        if not isinstance(resource, TemperatureFitProcessAdapter):
            raise RuntimeError(
                "leased temperature-fit resource has unexpected adapter type: "
                f"{type(resource).__name__}"
            )
        return resource.fit_temperature_laws(request)
    except TemperatureFitInputError as error:
        raise ToolException(f'温度拟合输入不符合要求：{error}。请修正输入或将任务升级待决；不得伪造拟合结果。') from error


# 只把显式输入合同错误交回模型；进程、通信与未知计算异常仍抛出。
fit_conductivity_temperature_law.handle_tool_error = True
