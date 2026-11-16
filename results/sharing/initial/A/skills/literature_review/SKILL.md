---
name: literature_review
description: 文献综述检索，根据研究主题系统检索钠离子电池相关文献并汇总
triggers:
  - 文献综述
  - 文献调研
  - 相关研究
  - 研究现状
workflow:
  steps:
    - id: understand_topic
      tool: load_skill
      required: false
      description: 加载本 SKILL.md 全文，确保遵守综述工作流
    - id: kg_explore
      tool: explore_entity_neighborhood
      required: false
      description: 若用户提到具体实体，先查 KG 获取相关实体与关系
    - id: retrieve_multiview
      tool: search_literature_multiview
      required: false
      description: 多角度检索主题文献（机理+材料+性能/条件）
    - id: retrieve_focused
      tool: search_literature_chunks
      required: false
      description: 若多视角检索不足，做聚焦补充检索
    - id: synthesize
      tool: synthesize_design_report
      required: false
      description: 生成带 chunk_id 引用的文献综述
---

## 行动与交付原则

主 Agent 根据用户目标、已有资料与证据缺口自主规划和选择行动；下文的方法和接口示例不是固定执行序列。
最终结论由主 Agent 直接撰写，说明配方或研究结论、来源、适用条件与限制。
不要求调用报告工具或填满固定模板；证据不足时可以保留缺口并明确结束。
`synthesize_design_report` 及其 claims、comparison_records 等参数说明仅用于显式需要旧版结构化输出的兼容调用，不能反过来规定普通研究的行动或结束时机。


# 文献综述

## 适用场景
用户需要了解某个主题的研究现状、相关文献或研究进展。

## 核心原则
- **所有结论必须标注来源**，格式：`[chunk_id]`
- 工具返回为空时，明确告知"未检索到相关文献"，**不自行编造**
- 检索不充分时换关键词继续，不要一次搜完就回答
- 综述类问题**不要**堆数据模式词；只有在需要具体数值时才加入 `mAh g`、`mS cm`、`cycles` 等

## 查询模板
需要检索时，根据主题和证据缺口选择英文查询，可参考以下角度，不要求固定数量：

- **机理角度**：`"{topic} mechanism sodium-ion battery electrolyte"`
- **材料角度**：`"{material} sodium-ion battery electrolyte performance"`
- **性能/条件角度**：`"{topic} low temperature cycling stability sodium-ion battery"`
- **对比角度**：`"{material_a} vs {material_b} sodium-ion battery electrolyte"`

参数建议：`top_k=10`，`per_query_top_k=8`，`max_per_paper=2`。

## 工作流

1. **理解主题**: 确认用户的研究主题和关注点（如特定材料、性能指标、机理等）
2. **图谱探索**: 需要实体关联或结构化线索时，可调用 `explore_entity_neighborhood`
3. **按需检索**: 先检查用户提供资料和本轮证据；缺少主题覆盖时可用多角度检索，缺少特定事实时可聚焦检索或阅读指定论文，不要求固定次数
4. **聚焦补充**: 若结果不足，换英文关键词再次检索；或调用 `search_literature_chunks` 做聚焦补充
5. **综合回答**: 主 Agent 直接形成与问题相符的综述，不需要调用报告工具
   - 区分已有共识和存在争议的观点
   - 每个论点标注 `[chunk_id]`
   - 文献不足的方向明确标注"文献数据有限"

## 禁止行为
- 禁止把多篇文献的性能数值迁移到未经验证的组合上
- 禁止输出"【推断】"句，无直接证据的陈述直接删除
- 禁止把综述中转述的原始文献结论伪装成已获取的一手文献

## 引用纪律
- 只能引用本次检索实际返回的文献
- 使用 `[chunk_id]` 格式引用，禁止改为 `[1]` 或 `[作者]`
