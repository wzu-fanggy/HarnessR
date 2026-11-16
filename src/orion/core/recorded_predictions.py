"""从本轮真实工具消息读取预测结果，独立于候选的推荐状态。"""
from __future__ import annotations

from orion.core.task_scope import current_task_messages

import json
from copy import deepcopy
from typing import Any, Mapping

from langchain_core.messages import AIMessage, ToolMessage
from orion.tools.electrolyte_design.models import ModelPropertyPrediction


def recorded_predictions(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """保留执行时的输入及返回；不从模型草稿或筛选参数推测计算结果。"""
    calls = {}
    records = []
    for message in current_task_messages(state):
        if isinstance(message, AIMessage):
            calls.update({call['id']: call for call in message.tool_calls or []})
            continue
        if not isinstance(message, ToolMessage) or message.name != 'predict_formulation_conductivity':
            continue
        call = calls.get(message.tool_call_id)
        if not call or call.get('name') != message.name or message.status == 'error':
            continue
        request = (call.get('args') or {}).get('request') or {}
        inputs = {str(c.get('candidate_id')): c for c in request.get('candidates') or []}
        try:
            payload = json.loads(message.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, list):
            continue
        for candidate in payload:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get('candidate_id') or '')
            if candidate_id not in inputs:
                continue
            for raw in candidate.get('model_predictions') or []:
                prediction = ModelPropertyPrediction.model_validate(raw)
                if prediction.status != 'predicted' or prediction.numeric_value is None:
                    continue
                records.append({'candidate_id': candidate_id, 'tool_name': message.name,
                    'tool_call_id': message.tool_call_id, 'candidate_input': deepcopy(inputs[candidate_id]),
                    'composition_input': deepcopy((request.get('composition_inputs') or {}).get(candidate_id) or {}),
                    'prediction': prediction.model_dump(mode='json')})
    return records


def prediction_text(record: dict[str, Any]) -> str:
    p = record['prediction']
    text = (f"候选 {record['candidate_id']} 的 {p['model_id']} 模型预测 {p['property_name']} 为 "
            f"{p['numeric_value']:.6g} {p['unit']}")
    if p.get('temperature_c') is not None:
        text += f"，温度 {p['temperature_c']:g} °C"
    text += '。'
    composition = record['composition_input']
    for key, label in [('molar_ratios', '输入摩尔比'), ('component_mmol', '输入用量（mmol）')]:
        if composition.get(key):
            text += label + '为 ' + ', '.join(f'{name}={value:.6g}' for name, value in composition[key].items()) + '。'
    if p.get('ensemble_std_log10') is not None:
        text += f"树间 log10 标准差为 {p['ensemble_std_log10']:.6g}。"
    if p.get('estimated_factor_1sigma') is not None:
        text += f"工具返回的 1σ 倍数为 {p['estimated_factor_1sigma']:.6g}。"
    return text + '以上为模型计算结果。'
