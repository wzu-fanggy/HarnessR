---
name: electrolyte_inverse_design
description: 钠离子电池电解液故障逆向设计与性能问题调整
triggers:
  - 逆向
  - 容量骤降
  - 失效
  - 如何调整
  - 为什么
  - 性能下降
method:
  objective: 建立可证伪的竞争假设与区分实验；完整配方只是可选 design 子任务
  evidence_gaps:
    - 故障边界或已知工况缺失
    - 假设没有支持、反证或缺失证据账本
    - 区分实验不能产生不同预期结果
  capabilities:
    - 机理与论文内检索
    - 结构化 KG 发现
    - competing-hypothesis 评估
    - 可选配方候选与约束筛选
  exit_conditions:
    - diagnosis_supported
    - diagnosis_hypothesis_only
    - insufficient_evidence
workflow:
  steps:
    - id: load_skill
      tool: load_skill
      required: false
      description: 加载本 SKILL.md 全文，确保遵守 inverse_design 工作流
    - id: retrieve
      tool: search_literature_multiview
      required: false
      description: 多角度检索故障机理与改善策略证据
    - id: retrieve_fallback
      tool: search_literature_chunks
      required: false
      description: 聚焦补充检索
    - id: retrieve_paper
      tool: search_paper_chunks
      required: false
      description: 对最相关单篇论文做有界聚焦检索
    - id: numeric
      tool: extract_numeric_properties
      required: false
      description: 涉及具体数值对比时抽取数值证据
    - id: formulate
      tool: query_formulation_space
      required: false
      description: 获取与故障症状对应的候选调整方案
    - id: screen
      tool: screen_formulations
      required: false
      description: 按结构化约束评估候选调整方案并给出 DecisionState
    - id: optimize
      tool: optimize_formulations
      required: false
      description: 旧调用方需要动作列表时翻译 DecisionState
    - id: report
      tool: synthesize_design_report
      required: false
      description: 编译最终逆向诊断报告与 TaskArtifact
---

## 行动与交付原则

主 Agent 根据用户目标、已有资料与证据缺口自主规划和选择行动；下文的方法和接口示例不是固定执行序列。
最终结论由主 Agent 直接撰写，说明配方或研究结论、来源、适用条件与限制。
不要求调用报告工具或填满固定模板；证据不足时可以保留缺口并明确结束。
`synthesize_design_report` 及其 claims、comparison_records 等参数说明仅用于显式需要旧版结构化输出的兼容调用，不能反过来规定普通研究的行动或结束时机。


# 电解液逆向设计 Skill

## 适用场景

- 电解液在特定工况下出现性能问题（如低温容量骤降、循环衰减）；
- 用户要求从溶剂/盐/添加剂角度反推调整方案。

## 工作流

1. 解析故障现象或目标性能；
2. 先检查已有材料，按缺口选择下列检索方向：
   - 角度 1：故障机理（如 "low temperature capacity fade hard carbon SEI"）；
   - 角度 2：溶剂/盐/添加剂改善策略；
   - 角度 3：替代体系或已有成功案例。
   - 参数建议：`top_k=10`，`per_query_top_k=8`，`max_per_paper=2`。
3. （可选）调用 `kg_query(query_type="what_improves", params={"target": "..."})` 反查关键材料。若已有证据包含明确 `paper_id`，且最相关论文缺少显式配方或故障工况证据，可优先对最相关论文调用 `search_paper_chunks(paper_id=..., query=..., top_k=8)`；`query` 用 4–8 个英文技术词描述缺失事实，不含作者名。将返回块追加到 `evidence_chunks`，不得跨论文拼接配方与性能；
4. 先建立至少两个竞争假设、各自的支持/反证/缺失证据、混杂因素和区分实验。
   任务确需新的候选调整方案且用户未提供可评估候选时，可调用 `query_formulation_space`
   获取候选调整方案，把故障症状、温度、正负极和安全限制转成：
   `{"request":{"task_type":"inverse_design","target_conditions":[故障对应工况...],"preferred_salts":[...],"preferred_solvents":[...],"preferred_additives":[...],"excluded_components":[...],"evidence_chunks": evidence_chunks,"max_candidates":3}}`；`evidence_chunks` 必须传入本轮已获取、与诊断和调整方向最相关的最多 12 个真实来源 chunk，不得让候选生成脱离本轮文献证据；保留工具原样返回的 `candidates`。
   如果候选 evidence 只有 `chunk_id` 而 `text/excerpt` 为空，先调用
   `enrich_formulation_evidence({"request":{"candidates": candidates}})`，并用返回候选
   替换原列表；证据已有原文时不得重复补全。
5. 需要对调整候选作结构化约束评估时，调用 `screen_formulations`：
   `{"request":{"candidates": candidates,"criteria":[原始故障/安全约束...],"target_conditions":[故障对应工况...]}}`；条件尽量写成可验证的电压、温度、循环次数、容量保持率、电导率、粘度或组分约束。保留工具原样返回的 `screening_results`、其中的 `criteria_evaluations` 与 `gaps`，不得手写分数、validations 或 scores。任何 `failed`、`insufficient_evidence` 或 `unsupported` 判定都不得在报告中表述为故障已改善。
6. 旧调用方需要动作列表时，可调用 `optimize_formulations`：
   `{"candidates": candidates,"screening_results": screening_results,"request":{"task_type":"inverse_design","objectives":[希望改善的指标...],"constraints":[原始温度、正极/负极和安全限制...]}}`。
   `objectives` 必须写故障希望改善的指标（如 `capacity retention`、`low temperature performance`），`constraints` 保留原始温度、正极/负极和安全限制。`screening_results` 必须原样传给优化工具；不得手写分数、validations 或 scores。不得同时传入 `validations` 或 `scores`。该兼容工具只翻译筛选后已经产生的 `DecisionState`，不会再按 score 独立裁决；若 proposal 为 `retrieve_more_evidence`，Agent 可继续补证、换候选，或保留缺口完成诊断报告。仅在继续争取完整配方推荐且证据或候选变化时重新筛选，不为诊断收尾强制补检索或筛选；未获准入的配方不得被推荐。
7. Agent 综合证据后直接给出诊断结论或待验证假设，说明缺口和可区分的验证方法。
   以下仅适用于显式请求旧结构化 artifact 的 `synthesize_design_report` 接口，不是普通研究的必经步骤。
   该接口使用 `diagnosis_supported`、`diagnosis_hypothesis_only` 或 `insufficient_evidence` 标签，参数示例：
   `{"request":{"task_type":"inverse_design","question":用户问题,"evidence_chunks": evidence_chunks,
   "candidates": candidates,"screening_results": screening_results,
   "hypotheses":[...], "confounders":[...], "discriminating_tests":[...]}}`。
   至少需要两个假设；**不要硬凑第三个假设**。每个假设必须显式记录支持、反证或缺失证据；
   缺乏直接证据的假设应写入 `missing_evidence` 而不是作为正式假设输出。
   区分实验必须覆盖至少两个假设，为它们给出不同的预期结果，并记录受控变量。自由文本测试列表不能通过门禁。
   若可选配方子任务为 `hold`，不得绕过筛选结论声称故障已经改善。配方 hold 不会抹掉
   已建立的诊断假设，但也不能把假设升级为已验证调整方案。
   为提升 `answer_relevancy`，请在 `request.direct_answer` 中提供 **1–2 句话**直接回答用户问题：
   - 第一句必须是对用户问题的直接回应（可改写原句），并给出核心结论；
   - 只写最精简的根因判断和条件化调整方向，不要展开机理、不要罗列假设/区分实验；
   - 必须包含问题关键词（如首圈库仑效率、SEI、溶剂、盐、添加剂、调整方向等）。
   **关键纪律**：完整配方属于可选子任务；仅在需要发现新候选或评估其约束时选择相应工具。
   在调用 `synthesize_design_report` 时，`request.direct_answer` 不能为空，否则报告工具应被拒绝或重新生成。
   报告将渲染为“直接回答”小节，避免 RAGAS 把后续结构化细节判为不相关。
   为提升 `claim_supported_ratio`，请在 `request.claims` 中提交**所有**被 `evidence_chunks`
   直接支持的事实声明（`claim_type="fact"`），每条包含 `text`、对应 `chunk_id` 和
   唯一逐字原文引文 `evidence_quote`（必须从该 chunk 的 `content` 中逐字截取，且只出现一次）。
   **禁止为凑数量编造 claim**；根据科学缺口决定是否补检索，选择结束时明确标注未解决限制，
   只提交已有证据支持的 claim。报告将单独渲染
   “已校验事实”小节，这些声明会显著提高事实支持率。
8. **根据实际问题选择有用的诊断内容，不要求固定章节**：
   - **直接回答**：报告开头用 1–2 句话直接回应用户问题（如“如何进行电解液逆向分析/调整”），给出核心结论和最精简的条件化调整方向；必须包含问题关键词（如首圈库仑效率、SEI、添加剂、溶剂、盐等），不要展开机理或罗列假设，再进入后续结构化小节。
   - **根因分析**：从溶剂、盐、添加剂三个角度分别说明；
   - **证据摘要**：每条陈述必须绑定 [chunk_id]；
   - **调整建议**：对应根因，每条建议标注证据等级；
   - **候选方案**：1–3 个，说明与证据的关联，不强求完整配方。

## 退出纪律

- 诊断任务不强制配方发现或筛选；已有候选、已有证据和实际缺口共同决定所需工具；
- 配方子任务存在时遵守其 `DecisionState`；`hold` 时不得绕过筛选结论声称故障已经改善；
- 区分实验计划不是实验结果，不能单独产生 `diagnosis_supported`。

## 诊断纪律

- 根因必须基于检索到的证据；
- 必须先给出根因，再给出调整建议，禁止只列建议不解释机理；
- 根因描述中必须包含便于指标识别的中英双语关键词，如"Na+ 去溶剂化/desolvation 缓慢/迟缓"、"盐析出/salt precipitation"、"SEI/界面恶化"等；
- 调整建议必须能从溶剂/盐/添加剂三个角度给出；
- 无直接证据的改善策略直接删除，禁止标注"推断"；
- 禁止把某条证据中的性能数值（如容量保持率、循环次数）迁移到未经验证的候选方案上。

## 引用纪律

同 `electrolyte_design`。

## KG 质量反馈

同 `electrolyte_design`：诊断过程中若发现 KG 实体标签错误、关系语义错误、数值异常、同名异义或证据与 KG 矛盾，**必须**调用 `flag_suspicious_kg_item` 记录，供后续离线人工审查。

## 知识沉淀

诊断流程结束后，调用 `reflect_now` 进行知识反思和 Wiki 沉淀，
将本次诊断发现的故障模式、缓解策略的有效性、检索词优化建议记录到 LLM Wiki，
供下次逆向诊断任务直接复用。
