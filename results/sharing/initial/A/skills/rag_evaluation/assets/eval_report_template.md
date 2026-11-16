# RAG / Agent 评估报告

## 评估配置

- 数据集：`{{dataset_path}}`
- 样本数：`{{num_samples}}`
- 评估时间：`{{eval_time}}`

## 指标汇总

| 指标 | 均值 | 备注 |
|------|------|------|
| faithfulness | {{faithfulness}} | 答案是否有证据支撑 |
| faithfulness_evidence | {{faithfulness_evidence}} | 证据引用准确性 |
| context_recall | {{context_recall}} | GT 关键信息被召回比例 |
| context_precision | {{context_precision}} | 召回上下文与问题相关度 |
| key_concept_recall | {{key_concept_recall}} | 关键概念召回 |

## 失败案例分析

{% for case in failure_cases %}
### 案例 {{loop.index}}
- 问题：{{case.question}}
- 失败指标：{{case.metric}}
- 诊断：{{case.diagnosis}}
{% endfor %}

## 改进建议

1. ...
2. ...
