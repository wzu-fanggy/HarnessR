"""等待同批工具全部结束后再传播异常，避免租约先于工作线程归还。"""
from dataclasses import dataclass

from langgraph.prebuilt import ToolNode


@dataclass
class _ToolFailure:
    error: BaseException


class LeaseSafeToolNode(ToolNode):
    """保持工具输入、结果和 Command 合并语义，只延后批次异常传播。

适配当前 LangGraph ToolNode 的单调用和合并扩展点；升级 LangGraph 时须运行
并行失败与取消合同测试。工具内部自行脱离调用生命周期的后台任务不在租约内。
"""

    async def _arun_one(self, *args, **kwargs):
        try:
            return await super()._arun_one(*args, **kwargs)
        except BaseException as error:
            return _ToolFailure(error)

    def _combine_tool_outputs(self, outputs, input_type):
        failures = [item.error for item in outputs if isinstance(item, _ToolFailure)]
        if failures:
            first = failures[0]
            for error in failures[1:]:
                first.add_note(f"another tool in the same batch failed: {error!r}")
            raise first
        return super()._combine_tool_outputs(outputs, input_type)
