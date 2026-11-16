---
name: electrolyte_design
description: 钠离子电池电解液配方正向设计与候选配方筛选
triggers:
  - 设计
  - 配方设计
  - 推荐电解液
  - 优化电解液
method:
  objective: Agent 在用户目标与真实证据下选择行动，工具检查提供确定性反馈，不替代 Agent 的语义判断
  evidence_gaps:
    - 候选组成或浓度缺失
    - 目标工况与性能没有同实验绑定
    - 硬约束缺少可验证原文
  capabilities:
    - 文献与论文内检索
    - KG 候选发现
    - 原文证据补全
    - 纯约束评估与有界重规划
  exit_conditions:
    - Agent 依据本轮证据形成结论，并说明已支持和未解决的目标
    - Agent 选择不推荐的受限结论，报告缺口和验证计划；不为结束而制造终态评估
workflow:
  steps:
    - id: retrieve
      tool: search_literature_multiview
      required: false
      description: 多角度背景检索，补齐候选、机理或替代体系证据
    - id: kg_structured
      tool: kg_query
      required: false
      description: 需要结构化材料、配方、条件或关系路径时查询领域知识图谱
    - id: retrieve_fallback
      tool: search_literature_chunks
      required: false
      description: 聚焦补充检索
    - id: resolve_paper
      tool: resolve_literature_papers
      required: false
      description: 用作者、题名、DOI 或用户给出的具体实验标识定位候选论文
    - id: retrieve_paper
      tool: search_paper_chunks
      required: false
      description: 已锁定具体论文时，在论文内部补取任务相关原文证据
    - id: numeric
      tool: extract_numeric_properties
      required: false
      description: 从证据块抽取数值属性
    - id: formulate
      tool: query_formulation_space
      required: false
      description: 检索配方相关原文材料，由 Agent 阅读并确认候选
    - id: enrich
      tool: enrich_formulation_evidence
      required: false
      description: 补齐候选 evidence 原文与元数据
    - id: conductivity_ml
      tool: predict_formulation_conductivity
      required: false
      description: 对具有明确摩尔组成且位于 NaPF6–EC/EMC/PC 适用域内的候选估计离子电导率；预测只作定量辅助，不替代文献证据
    - id: screen
      tool: screen_formulations
      required: false
      description: 按结构化 constraint_records 评估候选
    - id: optimize
      tool: optimize_formulations
      required: false
      description: 将 DecisionState 翻译为动作列表或推荐/暂停
    - id: report
      tool: synthesize_design_report
      required: false
      description: 编译最终设计报告与 TaskArtifact
---

# 电解液证据驱动研究

面向已有文献中的钠离子电池电解液候选；新组合只能作为明确标注的待验证假设。
主 Agent 根据问题、已有证据和本 Skill 自主决定下一步及何时结束。
不要求无条件调用所有工具；frontmatter 的 workflow 仅是兼容能力目录，不表示执行顺序。

## 明确要支持的结论

- 区分用户要比较已有候选、综述文献、按工况找配方，还是解释失效机制；已有完整候选可直接比较，综述不必发现或筛选配方。
- 保留用户给出的温度、电极、电池形式、电压、倍率/电流、性能指标及定性目标，只检查当前结论真正需要的条件。未要求的长循环或机理数据不自动成为阻断条件。
- 区分精确阈值、近似目标与“报告某指标”。缺少容差或数值判据时不自造阈值；温度 C 与倍率 C 按语境解释，确实不明时澄清，不能同时施加互斥要求。
- 近似目标保留原话，报告实际测试条件和偏差；其他温度、电极或电池形式的结果不冒充目标工况已验证。成本动机不等于价格优势。

## 根据缺口选择行动

- 已有材料足以支持有限结论时直接回答；无需为工具清单、候选数量或报告模板重复检索。
- 缺少候选或关系入口时可选 `kg_query`、`query_formulation_space`；后者默认返回原文材料，由你阅读并构造候选，不负责完整配方语义提取。也可直接阅读其他检索工具结果，无须经过候选生成工具。用户未限定的盐或溶剂不预先写成硬约束。
- 缺少相关文献时可选 `search_literature_chunks` / `search_literature_multiview`。用英文围绕实际工况与缺失事实查询，保留有辨识度的电极、数值、单位和配方标识；根据返回结果调整，不固定查询次数。
- 已知论文或配方标识时，可用 `resolve_literature_papers` 定位，再用 `search_paper_chunks` 追查组成、方法、正文或图注。向量检索未命中不等于论文未收录，不猜作者、题名或 DOI。
- 证据只有 ID 而无正文时，可选 `enrich_formulation_evidence`；已有正文不必重复补齐。图注只能补充明确对应的子图，不把相邻实验的条件移过来。
- `lookup_measurement` 只检索本地体系记录；`system_only` 和温差接近均不证明完整配方相同。空结果只表示本次库查询未命中。
- 一次筛选得到 `hold`、工具报错或某个模型超域后，可以补证、换候选、改用文献、重试适用工具，或说明无法解决的部分后结束。工具反馈不得自动替代科研规划。
- 方法陌生、证据广泛或冲突难以解释时增加研究深度；可拆成独立论文/候选核验的任务再委派子代理，传入本轮问题、待审结论及原文，要求返回来源、支持/反证和缺口。主 Agent 综合结果；首次遇见不强制启动深度研究，子代理共识不是实验验证。

## 核验配方—测试—结果关联

- 核对盐及各盐浓度、溶剂及比例基准、添加剂及用量；未知保持未知。“未提及添加剂”不等于“确认无添加剂”，不以字符串占位补齐字段。
- 相同论文、相同盐或溶剂集合都不足以建立归属。用原文配方标识、明确图表/方法链接及测试分支核对浓度、比例、温度、电极、电池形式、倍率、循环数和性能定义；不得跨论文拼接配方与性能。
- TD37/TD73 一类编号以原文定义为准，不能借另一比例的数值；容量、保持率、首圈效率和循环寿命分别表达。性能最好但组成缺失的分支不能借完整分支的组成来推荐。
- 同条件直接对比可支持该测试范围内的优劣；跨研究结果说明条件差异与混杂因素，不能无依据宣布最优。矛盾并列保留，可追查原文、测量定义与重复性；未消解时不选一个值当真。
- 文献实测、用户提供值、模型预测和待验证推断分别表达。引文逐字来自本轮可用原文，来源 ID 原样使用；引文存在不等于其语义支持断言。图谱和记忆不替代原文证据。

## 何时计算或结构化评估

- 仅在计算能解决当前问题且输入与方法适用时调用。现有电导率模型只覆盖 NaPF6–EC/EMC/PC、无添加剂且具有明确摩尔组成的候选；温度/组成范围以工具实际检查为准。
- 不将“1 M + 1:1”或未标基准的 `solvent_ratios` 自行换成包括盐的摩尔输入。`out_of_domain` / `insufficient_composition` 不能当成性能差或整个任务失败。
- `model_predictions` 与文献证据分开；预测电导率不能单独证明低温循环、高压稳定、SEI 或完整工况达标，模型离散度也不自动等于校准后的置信区间。
- 需要结构化约束评估时可用 `screen_formulations`，不得把返回的 `failed`、`insufficient_evidence` 或 `unsupported` 写成通过。证据或候选改变后，先重新评估再声称该接口准入通过；不为提交有限结论而强制评估。
- 显式使用结构化筛选、`optimize_formulations` 或旧版报告接口时，按需读取 [结构化评估与旧接口](references/structured_evaluation.md)。其中参数合同只约束所选接口，不强制普通研究调用它。

## 交付和经验

回答用户当前问题：给出已知配方组成、支持结论的实测条件与性能、来源，以及尚未证实的目标。
有充分证据时给出其适用范围内的候选建议；证据不足时说明还能判断什么、不能判断什么及最有价值的下一项核查。
推断可以给出，但标注依据和待验证性质；不能把拟议实验写成已取得的结果。
主 Agent 直接回答，不需要先调用报告工具。显式使用旧报告协议时仍保持正文、artifact 和来源一致。

不伪造引文、测试、计算结果或准入状态；预算/权限和当前任务边界继续由既有执行机制约束。
遇到可疑 KG 项可按权限用 `flag_suspicious_kg_item` 留线索，不自动矫正图谱。
经验仅记录带出处和适用条件的候选经验，不把自我反思直接当成科学事实，不自动修改代码或 Skill。

## 输运性质脚本入口

预测、批次评价、读取保存结果及跨温度比较有原生工具入口，按需读取[输运计算与批次合同](references/transport_tools.md)。较大候选空间可用generate_candidate_space的artifact模式保存，再用evaluate_candidate_batch评价、inspect_candidate_artifact读取。compare_conductivity_predictions核对实际保存记录；不能手工重标结果。根据科研目标选择操作，不要求固定顺序或额外调用。独立脚本保留离线兼容，不与原生计算重复运行。


同配方温度对比应引用各次计算的 `result_binding.record_id`，按需使用输运脚本的 `compare` 核对后计算比值。配方改变后不能把旧数值改名继承；缺少某温度计算就补算或说明缺口。`compared` 只说明该模型输入满足比较合同，不验证文献归属或真实电池性能。


批量提出或筛选候选时，可用输运脚本的generate/evaluate/inspect，或让原生生成工具返回artifact模式，按需读取[批处理用法](references/transport_tools.md)。预览不是推荐；先评价实际考察集合，再按明确指标说明取舍。引用生成数、成功预测数、未评价数和evaluation_id，有限采样中的点预测排名不能写成全空间或电池性能最优。

批处理说明用read_file读取Orion工作区内路径 `skills/electrolyte_design/references/transport_tools.md`；Markdown不是可执行脚本。Python入口给run_script的路径为 `electrolyte_design/scripts/evaluate_transport.py`。


## 配方计量求解

用`run_script`执行`electrolyte_design/scripts/solve_formulation.py`，输入只含组分ID、计量约束和归一化条件；输出组成、欠定范围、冲突或输入错误。新接口不传原文和文字假设，解释与来源留在任务记录中。按需用`read_file`读取`skills/electrolyte_design/references/formulation_stoichiometry.md`；物性由固定目录提供，组成已明确时无需重复求解。


## 组成输入检查

已有`check_formulation_feasibility`只核对组成完整性/一致性和参考物性。按需读取[输入与结果范围](references/composition_checks.md)；盐的绝对用量不能为相对溶剂比例定标。采用composition_status判断输入状态；feasible为null、物理可配制性未评价，不能把它当性能准入或淘汰结论。
