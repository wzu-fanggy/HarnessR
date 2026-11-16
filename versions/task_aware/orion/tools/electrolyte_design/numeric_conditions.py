"""从原文明确的成对数值和同一电池测试程序中保留条件对应关系。"""
from __future__ import annotations

import re
from orion.task_evidence.condition_extraction import extract_condition_scope
from orion.task_evidence.scientific_text import numeric_excerpt

_PERCENT = re.compile(r'(?<![\d.])(?P<value>\d+(?:\.\d+)?)\s*%')
_TEMP = re.compile(r'(?<![\d.])(?P<value>-?\d+(?:\.\d+)?)\s*(?:°\s*C|℃)', re.I)
_PAIR = re.compile(r'(?<![\d.])(?P<value>\d+(?:\.\d+)?)\s*%\s*'
                   r'\(\s*(?P<temperature>-?\d+(?:\.\d+)?)\s*(?:°\s*C|℃)\s*\)', re.I)
_DIRECT_CAPACITY = re.compile(
    r'\b(?:retain|maintain)(?:s|ed)?\s+(?P<value>\d+(?:\.\d+)?)\s*%\s+'
    r'(?:of\s+)?(?:(?:the|its)\s+)?(?:(?:room[ -]temperature|RT|original|initial)\s+)?'
    r'capacity\s+at\s+(?P<temperature>-?\d+(?:\.\d+)?)\s*(?:°\s*C|℃)', re.I)


def retention_temperature_pairs(text: str) -> list[dict]:
    """只接受保持率语境中的括号对应或明确的 respectively 等长列表。"""
    output, seen = [], set()
    for match in _PERCENT.finditer(text):
        quote = numeric_excerpt(text, match.start(), match.end())
        if quote in seen:
            continue
        seen.add(quote)
        direct = list(_DIRECT_CAPACITY.finditer(quote))
        if direct:
            output.extend({'value': float(p['value']), 'temperature_c': float(p['temperature']),
                           'excerpt': quote} for p in direct)
            continue
        if not re.search(r'capacity\s+retentions?\b|容量保持率', quote, re.I):
            continue
        pairs = list(_PAIR.finditer(quote))
        if pairs:
            entries = [(float(p['value']), float(p['temperature'])) for p in pairs]
        elif re.search(r'\brespectively\b|分别', quote, re.I):
            percentages = list(_PERCENT.finditer(quote))
            temperatures = list(_TEMP.finditer(quote))
            # 个数不同或列表交错时，不按距离猜测。室温文字不参与数值温度列表。
            if (len(percentages) < 2 or len(percentages) != len(temperatures)
                    or percentages[-1].end() > temperatures[0].start()):
                continue
            entries = [(float(p['value']), float(t['value']))
                       for p, t in zip(percentages, temperatures)]
        else:
            continue
        output.extend({'value': value, 'temperature_c': temperature, 'excerpt': quote}
                      for value, temperature in entries)
    return output


def same_paragraph_cell_context(text: str, excerpt: str, scope: dict) -> tuple[dict, list[dict]]:
    """把同段唯一明确电极对绑定到紧随其后的性能句，保留原句出处。

    不跨段、不传递温度/倍率，不根据两个分散的单电极叙述组装电池。
    """
    if scope.get('electrode') or not re.search(r'cycl|capacity|discharg|retention', excerpt, re.I):
        return dict(scope), []
    position = text.find(excerpt)
    if position < 0:
        return dict(scope), []
    left = text.rfind('\n\n', 0, position) + 2 if '\n\n' in text[:position] else 0
    prefix = text[left:position]
    found, seen = [], set()
    for match in re.finditer(r'\b(?:full[ -]cells?|half[ -]cells?|pouch[ -]cells?|coin[ -]cells?)\b', prefix, re.I):
        quote = numeric_excerpt(prefix, match.start(), match.end())
        if quote in seen:
            continue
        seen.add(quote)
        context = extract_condition_scope(quote)
        if context and context.electrode:
            found.append((context, quote))
    pairs = {context.electrode.casefold() for context, _ in found}
    shapes = {context.cell_configuration for context, _ in found if context.cell_configuration}
    if len(pairs) != 1 or len(shapes) > 1:
        return dict(scope), []
    context, quote = found[-1]
    if '/' not in context.electrode or context.cell_configuration == 'half-cell':
        return dict(scope), []
    if scope.get('cell_configuration') and scope['cell_configuration'] != context.cell_configuration:
        return dict(scope), []
    # 限于最近的电池叙述；另一个未解析的电池/电极实验也阻止条件延续。
    intervening = prefix[prefix.rfind(quote) + len(quote):]
    if re.search(r'\b(?:cells?|anodes?|cathodes?|electrodes?)\b', intervening, re.I):
        return dict(scope), []
    values = {'electrode': context.electrode}
    if context.cell_configuration:
        values['cell_configuration'] = context.cell_configuration
    return {**scope, **values}, [{'excerpt': quote, 'condition_scope': values,
                                  'relation': 'same_paragraph_explicit_cell'}]


def same_cell_voltage_protocol(text: str, excerpt: str, scope: dict) -> tuple[dict, list[dict]]:
    """只把显式全体充放电测试范围用于同段、同电极对的放电测试句。"""
    electrode = scope.get('electrode')
    if not electrode or not re.search(r'\bdischarge\s+(?:curves?|tests?)\b', excerpt, re.I):
        return dict(scope), []
    position = text.find(excerpt)
    if position < 0:
        return dict(scope), []
    # 空行/标题划定程序段落；不跨段落或其他原文块搬运条件。
    left = text.rfind('\n\n', 0, position) + 2 if '\n\n' in text[:position] else 0
    prefix = text[left:position]
    matches = list(re.finditer(r'\b(?:charge\s+and\s+discharge|charge[-–]discharge)\s+tests\b', prefix, re.I))
    found = []
    for match in matches:
        quote = numeric_excerpt(prefix, match.start(), match.end())
        conditions = extract_condition_scope(quote)
        if not conditions or not conditions.voltage_window_v or not conditions.electrode:
            continue
        if conditions.electrode.casefold() != str(electrode).casefold():
            continue
        # 特定温度、倍率或循环条件的范围不能作为整个测试程序的范围。
        if conditions.temperature_c is not None or conditions.rate or conditions.cycle_count:
            continue
        found.append((list(conditions.voltage_window_v), quote))
    if not found or len({tuple(v) for v, _ in found}) != 1:
        return dict(scope), []
    window, quote = found[-1]
    # 中间出现另一电压窗口或另一电极对时，不假定早先的范围持续适用。
    intermediate = prefix[prefix.rfind(quote) + len(quote):]
    check = extract_condition_scope(intermediate)
    if check and ((check.voltage_window_v and list(check.voltage_window_v) != window)
                  or (check.electrode and check.electrode.casefold() != str(electrode).casefold())):
        return dict(scope), []
    if scope.get('voltage_window_v') and list(scope['voltage_window_v']) != window:
        return dict(scope), []
    values = {**scope, 'voltage_window_v': window}
    links = [{'excerpt': quote,
        'condition_scope': {'electrode': electrode, 'voltage_window_v': window},
        'relation': 'same_cell_charge_discharge_protocol'}]
    # 同一测试程序的开头可明确给出软包外形，后文使用更一般的 full cell。
    # 只读取显式同电极对的句子；coin/half-cell 等相冲突的形式阻止继承。
    shapes = []
    seen = set()
    for match in re.finditer(r'\bpouch\b|\bcoin\b|\bhalf[ -]cell\b', prefix, re.I):
        shape_quote = numeric_excerpt(prefix, match.start(), match.end())
        if shape_quote in seen:
            continue
        seen.add(shape_quote)
        conditions = extract_condition_scope(shape_quote)
        if conditions and conditions.electrode and conditions.electrode.casefold() == str(electrode).casefold():
            shapes.append((conditions.cell_configuration, shape_quote))
    if shapes and {shape for shape, _ in shapes} == {'pouch'} and scope.get('cell_configuration') in {None, 'full-cell', 'pouch'}:
        values['cell_configuration'] = 'pouch'
        links.extend({'excerpt': shape_quote, 'condition_scope': {'electrode': electrode, 'cell_configuration': 'pouch'},
                      'relation': 'same_cell_charge_discharge_protocol'} for _, shape_quote in shapes)
    return values, links
