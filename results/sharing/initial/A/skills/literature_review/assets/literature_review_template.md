# {{topic}} 文献综述

## 研究背景

<!-- 用 1–2 段说明该主题在钠离子电池电解液领域的研究意义 -->

## 关键发现

### 1. {{finding_1_title}}
- {{finding_1_summary}} [{{chunk_id_1}}]

### 2. {{finding_2_title}}
- {{finding_2_summary}} [{{chunk_id_2}}]

## 共识与争议

- **已有共识**：...
- **存在争议**：...
- **文献数据有限**：...

## 引用来源

{% for chunk in chunks %}
- [{{chunk.chunk_id}}] {{chunk.paper_title}}{% if chunk.section_title %} — {{chunk.section_title}}{% endif %}
{% endfor %}
