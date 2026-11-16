---
name: electrolyte_screening
description: 钠离子电池电解液文献对比与模型性质初筛
triggers:
  - 作用是什么
  - 对比
  - 比较
  - 差异
  - 性能
  - 优缺点
  - FEC
  - NaPF6
  - NaFSI
  - 成膜
workflow:
  steps:
    - id: load_skill
      tool: load_skill
      required: false
      description: 加载本 SKILL.md 全文，确保遵守 screening 工作流
    - id: resolve_paper
      tool: resolve_literature_papers
      required: false
      description: 当问题指定作者、论文题名或 DOI 时，先解析 paper_id
    - id: retrieve
      tool: search_literature_multiview
      required: false
      description: 多角度检索文献证据；缺少相关证据时按需使用
    - id: retrieve_paper
      tool: search_paper_chunks
      required: false
      description: named-study 问题在已解析的单篇论文内部做一次有界聚焦检索
    - id: retrieve_focused
      tool: search_literature_chunks
      required: false
      description: 用题目中的实体、数值或原始短语做聚焦检索，核验首轮检索未覆盖的直接实验事实
    - id: numeric
      tool: extract_numeric_properties
      required: false
      description: 数值证据未结构化且需要精确比较时使用
    - id: report
      tool: synthesize_design_report
      required: false
      description: 生成带 chunk_id 引用的简洁回答
---

## 行动与交付原则

主 Agent 根据用户目标、已有资料与证据缺口自主规划和选择行动；下文的方法和接口示例不是固定执行序列。
最终结论由主 Agent 直接撰写，说明配方或研究结论、来源、适用条件与限制。
不要求调用报告工具或填满固定模板；证据不足时可以保留缺口并明确结束。
`synthesize_design_report` 及其 claims、comparison_records 等参数说明仅用于显式需要旧版结构化输出的兼容调用，不能反过来规定普通研究的行动或结束时机。


# 电解液 Screening Skill

## 适用场景

- 解释某种组分/添加剂的作用（如"FEC 的作用"）
- 对比两种或多种材料/体系（如"NaPF6 vs NaFSI"）
- 回答事实性、机理性问题
- 对用户提供的候选进行模型性质初筛，明确计算指标、组成与温度

## 模型性质初筛

当任务要求用预测模型筛选既有配方时，文献解释研究背景，模型计算评价候选。
两类来源分别保存。不要把预测值、模型排序或模型成员统计引用为某篇论文的实验结果，
也不要为性质预测编造电极、电压、倍率或循环数。

- 先明确候选组成、比例基准、目标温度及用户指定的筛选指标。已有候选保持身份一致，
  不自行增加配方或修改比例。
- 用现有性质预测能力获得目标温度下的计算结果。可在一次单温度请求中计算多种候选；
  由任务需要决定调用，不规定检索与计算的交替顺序。
- 未给出最低电导率要求时，不自行设置 1 mS/cm 等阈值。对于“目标温度范围内最低
  电导率尽可能高”的任务，可直接比较单点预测，不必调用要求阈值的集成排序工具。
- 编译报告时传 `task_type="candidate_screening"`、候选清单及
  `model_comparison={"candidate_ids":[...],"temperatures_c":[...],
  "objective":"maximise_minimum_conductivity","priority_count":1}`。若用户要求前 N 种优先候选，
  必须把 `priority_count` 设为 N（例如要求四种则传 4），并按筛选指标列出该数量的候选；
  并列末位可一同保留。此字段仅控制报告列出多少个优先候选，不改变预测模型或实验事实。
  编译器读取本轮工具消息，不在
  `claims` 内复制预测数值或给它们拼接文献 chunk_id。未完成计算应继续补齐相关温度，
  不用已知答案填补工具返回。
  用户要求比较参照温度时，把该温度传入 `reference_temperature_c`，不能只在文字建议中
  提及参照结果；`temperatures_c` 应只包含目标筛选温度，单独传入的参照温度不改变初筛指标。
  提交前对照用户问题检查候选、目标
  温度和参照温度是否完整，不能为了修复其他字段错误而省略已要求的比较。
- 用户要求文献解释时，只把直接相关且来源完整的背景事实作为 `claim_type="fact"` 放入
  `claims`，逐条附真实 `chunk_id` 与完整、有边界的逐字 `evidence_quote`。不要引用截在
  冠词、形容词或句中连词后的不完整摘句；不能仅凭一句关于容量的数据扩展出未获证实的
  电导率—电池表现因果关系。模型排序由编译器使用已执行预测计算，不需要用文献
  `claim_type="comparison"` 代替模型排序。这类非 fact 的文献声明在模型初筛编译时
  不会自动用于支持配方优先级。`next_steps` 必须明确给出至少两项尚未执行的具体验证，
  包括目标配方、温度下的独立电导率复测，以及在相同电极、电芯类型、倍率和电压范围下
  的实际容量、循环或阻抗比较；不能把建议写成已做实验。若模型遗漏 `next_steps`，
  报告编译器只会补充标明来源的通用复测项目，不算 Agent 独立设计的方案。
  这些背景事实并未被用来证明给定候选的电池性能，不需为它们编造目标电极或实验工况。
  工具接口公开了请求字段结构，不要添加结构以外的字段。
- `model_assessed` 表示已按显式指标完成模型初筛。展示具体候选、组成、温度、数值和
  选择理由，区分预测与实测；提出相应实验验证建议。这个状态不表示电池循环性能已被验证。

下方文献工作流适用于从论文实验数据得出事实和实验比较的任务。模型初筛中的背景
检索按实际证据缺口安排，不为凑文献数或操作数追加查询。

## 工作流

0. **论文身份解析**：如果问题指定的研究尚无可靠 paper_id，可调用
   `resolve_literature_papers`，只传身份标识本身
   （例如 `Fondard`），不要把整段问题当成身份查询。选择题名、作者和主题均
   与问题一致的首位结果，并记录其 `paper_id`。随后所有
   `search_literature_multiview` 调用都必须传入 `paper_ids=[该 paper_id]`，
   内容 query 去掉作者名，改用盐、溶剂、添加剂、循环次数、容量保持率或原文短语。
   只有 resolver 与限定论文检索都没有结果时，才能说明当前语料未找到该研究；
   禁止仅凭普通向量检索失败就声称论文未收录。
   已有论文原文可直接核验；仍有明确缺口时，可调用
   `search_paper_chunks(paper_id=..., query=..., top_k=8)` 做论文内聚焦。
   该能力与聚焦检索按实际缺口选择，
   query 只保留 4–8 个英文技术词，例如
   `"EC:DMC HC Na voltage cycles rate"`；不要堆叠完整句子、作者名或大量近义词。
   不为满足工作流清单重复读取已有材料。
   证据已覆盖组分、工况和指标时，由 Agent 判断是否需要数值抽取或收尾；
   禁止为了补齐无关细节反复改写查询，也禁止用 `read_file` 绕过检索访问原文路径。
   若仍缺少明确事实，可在执行预算内继续核验；选择收尾时须把缺口标为未验证。

1. **检索**：已有材料不足时，按缺口选择多角度、聚焦或论文内检索。
   查询模板（按问题类型选择，全部用英文）：
   - **机理解释**（如 FEC 的作用）:
     - `"FEC SEI film formation mechanism sodium ion battery"`
     - `"FEC reductive decomposition sodium ion battery"`
     - `"FEC cycling performance sodium ion battery"`
   - **材料对比**（如 NaPF6 vs NaFSI）:
     - `"NaPF6 sodium salt carbonate electrolyte common use"`
     - `"NaFSI aluminum corrosion sodium ion battery carbonate"`
     - `"NaPF6 NaFSI dual salt low temperature performance sodium ion battery"`
     - `"FSI anion solvation structure Na+ compared PF6"`
     - 缺少直接实验对比时，可用包含两种材料及工况/指标的查询，例如 `"NaFSI-E NaPF6-E -10°C capacity retention cycling comparison"`。
   - **性能/属性查询**:
     - `"{material} ionic conductivity low temperature sodium ion battery"`
     - `"{material} cycling stability sodium ion battery"`
   - 参数建议：`top_k=12`，`per_query_top_k=10`，`max_per_paper=2`。
   - 缺少直接对比证据时，可继续补检索、核验已有材料，或明确报告无法比较；不得编造对比结论。
2. **数值抽取**：需要结构化数值且已有材料未提供时，可调用 `extract_numeric_properties`；
3. **综合**：先按用户任务选择 canonical `task_type`：
   - 单一材料作用、事实、机理或研究现状：`evidence_synthesis`；
   - 用户明确给出两个及以上候选并要求比较/筛选：`candidate_screening`。
   仅在显式需要旧结构化 artifact、选择调用 `synthesize_design_report` 时，不得再传旧的 `screening` 标签。候选比较必须传入
   `comparison_axes`，并为每个候选传入一条 `comparison_records`：
   `{"candidate_id":"...","conditions":{"temperature_c":...,"electrode":"...",
   "voltage_window_v":[lower,upper],"rate":"...","cycle_count":...,
   "measurement_definition":"..."},"evidence_ids":["实际 chunk_id",...]}`。
   六类条件必须来自本轮原文证据，不能用模型常识补齐。报告编译器会确定性比较全部字段：
   缺字段或 evidence id 不属于本轮 ledger 时返回 `insufficient_evidence`，字段冲突时返回
   `not_comparable`，全部一致且存在通过逐字引文校验的 `comparison`/`ranking` claim
   时才允许 `compared`。旧 `condition_comparable` 布尔值不再具有准入权限；
   要输出可靠对比状态 `compared`，除 `comparison_records` 外，还必须在 `claims` 中提交至少一条
   `claim_type="comparison"` 或 `claim_type="ranking"` 的声明，直接回答“谁更好/有何不同”。
   每条 claim 必须包含 `text`（结论句，内嵌 `[chunk_id]`）、`chunk_id`、
   `evidence_quote`（逐字原文片段，≥12 字符）。示例：
   `{"claim_type":"comparison","text":"在 Fondard 的 NaPF6/EC:DMC 硬碳体系中，3% FEC 在 130 周后容量保持率为 94%，循环稳定性优于 DMCF [0053_paper4_fondard_sei::sec5::child2]","chunk_id":"0053_paper4_fondard_sei::sec5::child2","evidence_quote":"65%, 72% and 94% capacity retention was measured after 130 cycles with 0%, 1.5% or 3% FEC, respectively"}`
   只有 fact 或证据不足时，可明确说明当前无法完成可靠对比，不为指标编造 comparison 声明。
4. **输出**：直接回答问题，列出必要证据及限制，每条事实带 `[chunk_id]`。
   证据数量与来源范围由问题决定，不设置最低篇数或固定字数。
   指定论文的结论须绑定目标 paper_id；跨论文补充材料单独标明，不能拼接成同一实验。

## 交付纪律

- Agent 决定收尾时直接回答；不以检索次数或某个工具是否调用过代替证据判断。
- 编译失败时阅读反馈并选择修正或说明限制，不能冒充已交付。成功提交后保持正文、artifact 和来源一致。

## 禁止行为

- 用户已有候选时不必重新发现候选；需要结构化约束评估时可选择筛选工具，不为清单追加设计流程。
- KG 可用于定位材料和结构化线索；实验性能对比结论仍须有对应文献原文及可比测试条件支持。
- 文献事实问答不自行生成新的候选配方；模型初筛可以比较用户给定的候选。
- 文献事实问答不扩展成配方设计任务；模型初筛可提出有计算依据的测试优先级和验证建议。
- 禁止把"对比两种盐"改写成"推荐一套双盐完整配方"；
- 禁止输出"【推断】"句，无直接证据的陈述直接删除。

## 引用纪律

- 只能引用本次检索实际返回的文献；
- 文献数值、配方、性能值必须能在返回 chunk 原文中找到；预测值绑定实际工具计算。
- 使用 `[chunk_id]` 格式引用，禁止改为 `[1]` 或 `[作者]`。

## KG 质量反馈

同 `electrolyte_design`：筛选过程中若发现 KG 实体标签错误、关系语义错误、数值异常、同名异义或证据与 KG 矛盾，**必须**调用 `flag_suspicious_kg_item` 记录，供后续离线人工审查。

## 知识沉淀

筛选流程结束后，调用 `reflect_now` 进行知识反思和 Wiki 沉淀，
将本次筛选发现的有效对比维度、检索关键词、被否决的候选理由记录到 LLM Wiki，
供下次相似筛选问题直接复用。


## 配方计量求解

用`run_script`执行`electrolyte_design/scripts/solve_formulation.py`，输入只含组分ID、计量约束和归一化条件；输出组成、欠定范围、冲突或输入错误。新接口不传原文和文字假设，解释与来源留在任务记录中。按需用`read_file`读取`skills/electrolyte_design/references/formulation_stoichiometry.md`；物性由固定目录提供，组成已明确时无需重复求解。
