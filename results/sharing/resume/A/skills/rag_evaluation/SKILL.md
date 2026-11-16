---
name: rag_evaluation
description: 自动评估 RAG 召回质量、知识图谱查询质量和 agent 设计任务表现，输出评估报告
triggers:
  - 评估
  - 测试召回
  - RAG评估
  - 测试效果
  - 跑baseline
workflow:
  steps:
    - id: load_skill
      tool: load_skill
      required: true
      description: 加载本 SKILL.md 全文，确保遵守评估工作流
    - id: build_or_load_dataset
      tool: run_script
      required: false
      description: 若评估数据集不存在，运行 scripts/build_baseline_dataset.py 生成
    - id: run_eval
      tool: run_script
      required: true
      description: 运行评估脚本（如 scripts/run_baseline_eval.py）
    - id: report
      tool: synthesize_design_report
      required: false
      description: 生成带指标和诊断的评估报告
---

# RAG / Agent 评估 Skill

## 适用场景
用户想评估当前系统在电解液辅助设计/筛选场景下的检索质量、图谱查询质量或端到端任务表现。

## 数据来源
向量库和知识图谱的数据来源于 **Web of Science** 核心合集，聚焦钠离子电池硬碳负极低温电解液方向。

## 可用工具
- `search_literature_chunks(query, top_k, mode, sections)` — 向量+KG混合检索，返回文献 chunk
- `search_literature_multiview(queries, ...)` — 多查询融合检索
- `explore_entity_neighborhood(entities, ...)` — KG 子图检索
- `kg_query(query_type, params)` — 结构化图谱查询
- `extract_numeric_properties(evidence_chunks, ...)` — 从 chunk 中抽取数值属性
- `run_script(path, args)` — 运行评估脚本

## 评估类型

### 1. 检索召回评估
使用 `evaluation/baseline_v2.jsonl` 或自定义问题集，对每个问题：
1. 调用 `search_literature_chunks` / `search_literature_multiview`
2. 检查 GT 锚定文献是否进入 top-k
3. 记录 context_recall、context_precision、key_concept_recall 等指标

### 2. 图谱查询评估
使用 `evaluation/kg_baseline.jsonl` 或自定义查询：
1. 调用 `kg_query` / `explore_entity_neighborhood`
2. 检查返回的实体/关系是否与预期一致
3. 记录 success_rate、覆盖率等指标

### 3. 端到端设计任务评估
使用 `evaluation/baseline_v2.jsonl`：
1. 让 agent 执行完整设计/筛选流程
2. 用 `faithfulness`、`faithfulness_evidence`、`context_recall`、`context_precision` 等指标评估
3. 记录失败模式和改进建议

## 输出格式

```markdown
## 评估报告

### 指标汇总
| 指标 | 值 |
|------|-----|
| faithfulness | X.XX |
| context_recall | X.XX |
| context_precision | X.XX |

### 失败案例分析
- 案例 1: ...

### 改进建议
1. ...
```

## 注意事项
- 不要为了"通过测试"而放宽评判标准
- 最终报告要坦诚指出问题
- 评估时除了准确性，还要关注**对设计决策的支撑程度**
