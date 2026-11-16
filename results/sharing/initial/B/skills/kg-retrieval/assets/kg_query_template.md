# KG 查询计划

## 用户意图

{{user_intent}}

## 提取实体

{% for entity in entities %}
- **{{entity.role}}**: `{{entity.name}}` (别名: {{entity.aliases | join(", ")}})
{% endfor %}

## 查询策略

| 步骤 | 工具 | 参数 | 目的 |
|------|------|------|------|
| 1 | `{{tool_1}}` | `{{params_1}}` | {{purpose_1}} |
| 2 | `{{tool_2}}` | `{{params_2}}` | {{purpose_2}} |

## 结果整合

- 结构化关系：...
- 文本证据：...
- 数值证据：...
