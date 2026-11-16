from __future__ import annotations



import json

import re

from typing import Any



from orion.evolution.eval.base import EvalResult, Evaluator



try:

    from orion.llms import main_model

except Exception:  # pragma: no cover - allows importing this module without optional LLM deps

    main_model = None





class ElectrolyteDesignEvaluator(Evaluator):

    """Hybrid evaluator: rule-based dimensions + LLM-as-judge for qualitative ones."""



    DIMENSION_WEIGHTS = {

        "evidence_coverage": 0.25,

        "formulation_completeness": 0.20,

        "confidence_calibration": 0.20,

        "hallucination_free": 0.15,

        "tool_workflow": 0.10,

        "citation_format": 0.10,

    }



    async def evaluate(self, run_state: dict[str, Any]) -> EvalResult:

        content = self._extract_final_content(run_state)

        tool_calls = run_state.get("tool_calls", [])



        dims: dict[str, float] = {}

        dims["citation_format"] = self._check_citation_format(content)

        dims["hallucination_free"] = self._check_hallucination(content, tool_calls)

        dims["tool_workflow"] = self._check_tool_workflow(tool_calls)



        # LLM-as-judge for remaining dimensions

        llm_dims = await self._llm_judge(content, tool_calls)

        dims.update(llm_dims)



        score = sum(dims.get(k, 0.0) * w for k, w in self.DIMENSION_WEIGHTS.items())

        return EvalResult(

            score=round(score, 3),

            dimensions=dims,

            reasoning=f"dimensions={dims}",

            passed=score >= 0.6,

        )



    def _extract_final_content(self, run_state: dict[str, Any]) -> str:

        messages = run_state.get("messages", [])

        for m in reversed(messages):

            if isinstance(m, dict) and m.get("role") == "assistant":

                return str(m.get("content", ""))

            if hasattr(m, "content"):

                return str(m.content)

        return ""



    def _check_citation_format(self, content: str) -> float:

        citations = re.findall(r"\[\d+\]", content)

        return 1.0 if len(citations) >= 2 else 0.5 if citations else 0.0



    def _check_hallucination(self, content: str, tool_calls: list) -> float:

        # Simple heuristic: if tool_calls is empty but content claims numbers, penalize

        if not tool_calls and re.search(r"\d+\s*(mAh/g|°C|%)", content):

            return 0.0

        return 1.0



    def _check_tool_workflow(self, tool_calls: list) -> float:

        names = [

            t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")

            for t in tool_calls

        ]

        required = ["search_literature_chunks", "query_formulation_space"]

        hits = sum(1 for r in required if any(r in n for n in names))

        return hits / len(required)



    async def _llm_judge(self, content: str, tool_calls: list) -> dict[str, float]:

        if main_model is None:

            return {

                "evidence_coverage": 0.5,

                "formulation_completeness": 0.5,

                "confidence_calibration": 0.5,

            }

        prompt = f"""你是一位严格的评审。请对以下电解液设计回答的三个维度分别给出 0-1 分数（保留两位小数），只返回 JSON：



{content[:2000]}



维度：

- evidence_coverage: 是否覆盖了检索到的关键证据

- formulation_completeness: 是否给出完整候选方案（盐/溶剂/添加剂）

- confidence_calibration: 是否正确区分强/中/弱证据，不过度承诺



返回格式：{{"evidence_coverage": 0.8, "formulation_completeness": 0.7, "confidence_calibration": 0.9}}"""

        try:

            response = await main_model.ainvoke([("human", prompt)])

            text = response.content if hasattr(response, "content") else str(response)

            # Be tolerant of markdown code fences around the JSON.

            if "```" in text:

                text = text.split("```")[1].lstrip("json").strip()

            return json.loads(text)

        except Exception:

            return {

                "evidence_coverage": 0.5,

                "formulation_completeness": 0.5,

                "confidence_calibration": 0.5,

            }

