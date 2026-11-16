# 结构化评估与旧接口

仅在 Agent 选择结构化筛选或兼容报告接口时读取。本文件不规定检索、筛选、计算、报告的执行顺序。
普通候选比较、文献综述或有限回答无需构造这里的参数。科学语义仍由 Agent 结合原文判断。

## 结构化条件、断言和共同测试上下文

以下为 `screen_formulations` 的约束表达细节；示例中的盐、溶剂、电极和数值只适用于示例请求，不自动加到用户任务。

1. 解析用户目标与约束；把用户要求的每项条件写成结构化 `constraint_records`，例如：
   不得把整段用户问题放入 `target_conditions`，再把旧文本检查识别到的部分词语当成全部要求。
   推荐前逐项核对用户要求与检查条件，避免漏掉正极、负极、电池形式、温度、倍率或指标定义。
   检索前先解释用户给出的数值条件：它描述什么物理量、单位如何理解、哪些信息尚不明确。
   这一解释依据用户描述的工作场景，在检索前定下，不能由后来检索到的论文或知识图谱记录
   反过来改写；其它论文里恰好出现同样的 “N C” 写法，只说明那些论文的用法，不能用来判断
   用户这句话。
   温度与倍率的单位必须分清。省略度符号的 C 不能直接视为充放电倍率。
   按它在句中的位置判断：直接修饰电芯、电池或其使用环境时（如 “a pouch cell at -30 C”、
   “a full cell around 50 C”），是温度；只有与充放电动作或倍率词连用时
   （如 “discharged at 1 C”、“cycled at 0.5 C”、“1 C rate”），才是倍率。
   按上下文确定一种解释后，就用这一种建立约束，
   不因缺少度符号再补一条互斥的倍率或温度约束，也不因检索结果改回另一种解释。
   检索查询和筛选断言须明确写出所采用的
   物理量与单位。只有上下文确实无法确定物理量时，才同时列出两种解释并说明需要澄清，
   此时暂缓推荐，而不是把两种解释同时写成两条互斥的数值约束。
   接近目标温度的文献结果应保留实测温度，不能改写为目标温度。
   硬约束来自用户问题，不得自行加上某一正极、软包形式、长循环寿命或数值阈值。
   在若干可替代选项之间尚未确定时（例如某类溶剂或某类盐有多种候选），不得把它们
   同时写成必需的 `required_component`；每个选项应作为独立候选分别比较，不叠加互斥的必需组分。
   近似目标也必须进入筛选，不得因用户没有给出容差而省略。保留用户对目标的原话，
   用现有 `evidence_claim` 表达候选须有相应工况下的直接测试依据，而非把近似目标
   改成精确相等、单边不等式或自行设定容差。逐候选判断时，`supported` 须引用该
   候选的实际测试条件与性能原句，`basis` 写清实测条件与目标的差别及其关联依据。
   只有其他工况的结果、测试建议或机理预测时，不能据此判近似工况要求已满足。
   没有相应测试依据就保留 `not_found` 并补证，不能删掉该要求后批准推荐，再把它
   放进 `next_steps`。有邻近条件下的直接依据时，在报告中明确实际条件与目标的
   差别，精确目标下的后续验证可以另列。电极、电池形式和其他明确要求仍须检查。
   联合条件也须保留近似目标的含义：不得把近似值写成 `target_context` 中要求
   精确匹配的数值。近似工况及其电极、全电池归属可写在同一 `evidence_claim`
   中，引用实际共同测试的结果；组分身份等与工况无关的要求不并入该测试条件组。
   “低成本”等定性目标须区分文献提出的成本动机与已量化的价格优势；未提供价格数据
   时不得称已验证低成本。数值明确的用户要求仍按原值建立硬约束，不得因此放宽。
   无法从问题得出数值判据的定性目标，保留为选材目标，在报告中说明依据和待验证
   条件。定性或近似目标通过断言及其原文依据判断，不变成必须命中字面词的关键词条件。
   这不允许省略用户明确给出的电极、电池形式、精确温度或倍率要求。
   当用户目标包含某项性能指标（如 capacity retention relative to room-temperature
   capacity、离子电导率、倍率性能、首圈库仑效率），先分清用户给的是**数值阈值**还是
   **只要报告**：给了明确数值时写成对应数值约束——保持率用 `minimum_retention`、
   电导率用 `minimum_conductivity`、倍率用 `c_rate`，`value` 取用户给出的数值；
   只要求“报告该指标”“有无该结果”而未给数值时，不得把阈值填 0 了事——`value: 0`
   会让任何数字都通过，等于没有检查。这类“须被报告/须成立”的指标改用
   `kind: "evidence_claim"`（见下），把该指标写成一句断言。
   `field_presence` 只用于检查候选的配方字段（`concentration_mol_l`、
   `solvent_ratios`、`additives` 等）是否显式填写，绝不用于表达“某项性能必须被报告”。
   禁止把“capacity retention … must be reported”写成
   `"kind": "field_presence", "field": "additives"` 这类把性能指标错配到配方字段的约束。
   ```json
   {
     "request": {
       "candidates": candidates,
       "constraint_records": [
         {
           "constraint_id": "temp-low",
           "description": "operating temperature ≤ -20°C",
           "kind": "temperature",
           "operator": "<=",
           "value": -20,
           "unit": "°C",
           "required_evidence_scope": "candidate_same_paper",
           "can_replan": true,
           "search_hint": "find low-temperature cycling evidence"
         },
         {
           "constraint_id": "electrode-hard-carbon",
           "description": "hard carbon anode",
           "kind": "electrode_material",
           "evidence_terms": ["hard carbon", "hard-carbon"],
           "required_evidence_scope": "candidate_same_paper",
           "can_replan": true,
           "search_hint": "find hard carbon anode evidence"
         },
         {
           "constraint_id": "cell-half",
           "description": "sodium half-cell configuration",
           "kind": "cell_configuration",
           "evidence_terms": ["half-cell", "half cell", "Na||HC", "HC||Na", "pure Na"],
           "required_evidence_scope": "candidate_same_paper",
           "can_replan": true,
           "search_hint": "find half-cell configuration evidence"
         },
         {
           "constraint_id": "field-concentration",
           "description": "explicit salt concentration must be present",
           "kind": "field_presence",
           "field": "concentration_mol_l",
           "required_evidence_scope": "candidate_same_paper",
           "can_replan": false
         },
         {
           "constraint_id": "salt-napf6",
           "description": "NaPF6 as salt",
           "kind": "required_component",
           "value": "NaPF6",
           "required_evidence_scope": "candidate_same_paper",
           "can_replan": true,
           "search_hint": "find NaPF6 electrolyte evidence"
         },
         {
           "constraint_id": "solvent-thf",
           "description": "THF as solvent",
           "kind": "required_component",
           "value": "THF",
           "required_evidence_scope": "candidate_same_paper",
           "can_replan": true,
           "search_hint": "find THF solvent evidence"
         }
       ]
     }
   }
   ```
   定性或断言型要求（如“无集流体/壳体腐蚀被观测到”“无析钠/无枝晶”“提高首圈库仑效率”
   “该添加剂抑制产气”）不得写成 `mechanism_keyword` 关键词表——关键词命中分不清
   “发生腐蚀”与“未发生腐蚀”。改用 `kind: "evidence_claim"`：在 `claim` 里用一句
   用户原话写出对单个候选须成立的断言，在 `claim_verdicts` 里**逐候选**给出判定。
   语义判断（这段原文是否为该候选建立或推翻该断言）由你负责；程序只做两件可机械
   核验的事：判定须指向本次请求的候选，`quote` 须在该候选证据正文中逐字存在。
   因此措辞、否定、同义称呼的变化都不影响判定，你只需如实读原文：
   - `supported`：有原文明确为该候选建立断言，附 `chunk_id` 与逐字 `quote`；
   - `refuted`：有原文明确反驳（如原文说“会持续腐蚀铝箔”），附 `chunk_id` 与逐字 `quote`；
   - `not_found`：检索到的证据未直接提及，如实填写，由补证机制驱动下一轮检索。
   背景介绍、他人体系的结果、无实验支撑的预测都不算 `supported`。例如：
   ```json
   {
     "constraint_id": "no-corrosion",
     "description": "no current-collector or case corrosion observed",
     "kind": "evidence_claim",
     "claim": "该候选电解液不腐蚀铝箔或钢壳",
     "claim_verdicts": [
       {
         "candidate_id": "candidate-1",
         "verdict": "refuted",
         "chunk_id": "fan2024::sec3",
         "quote": "the single NaFSI system can continuously corrode aluminum foil and stainless steel sheets",
         "basis": "原文明确说明单盐 NaFSI 会持续腐蚀铝箔与不锈钢，与该断言相反。"
       }
     ],
     "can_replan": true,
     "search_hint": "find corrosion test evidence for the candidate electrolyte"
   }
   ```
   `evidence_claim` 不得把 `required_evidence_scope` 设为 `component` 或 `background`：
   断言必须由候选自身绑定的证据支持，不能借用别篇论文或背景段落的措辞。
   给出 `supported` 前，逐句核对引文是否真正支持所写断言。引文必须从指定
   `chunk_id` 返回的原文复制，不能把题名、摘要或另一块文字放在该块的标识下。
   配方、测试条件和性能必须对应同一候选；一种优点不能替代另一项目标的证据。
   成本动机与价格比较分别表述，安全性或无氟组成不能自行推导出成本优势。
   `basis` 说明原句与断言的推理关系，不得以重复断言代替理由。程序核验引文
   存在及归属，不代替科学含义判断；证据未支持完整断言时应补证或保留 `not_found`。
   `criteria`/`target_conditions` 仅用于兼容旧调用方；新设计任务必须优先使用
   `constraint_records`，避免 evaluator 再从自由文本猜测硬约束。

   `required_evidence_kinds` 描述的是**支持记录本身的语义角色**，不是实验条件元数据。
   数值性能约束通常只要求 `property`；温度、电极、循环数等共同实例化条件应由该
   observation 的 `condition_scope` 和 `joint_context_requirements` 表达。除非硬约束本身
   确实要求一条独立的 condition 记录，否则不要把同一数值约束写成
   `required_evidence_kinds=["property", "condition"]`，否则会把“带条件上下文的性能观测”
   错当成缺少第二种证据角色。

   当两个及以上硬约束必须在**同一个目标实验上下文**中同时成立时，还必须显式写
   `joint_context_requirements`。分别在多个倍率测试的要求，按倍率分别建立条件组，
   每组的 `target_context.rate` 只填一个倍率。不能把 `0.5 C and 1 C` 当成一次测试的倍率。
   同一论文的不同盐浓度或溶剂比例必须分别对应其性能段，不能因来自同篇论文就互换结果。
   `target_context.cell_configuration` 使用条件检查器的外形名称 `pouch` 或 `coin`，
   不拼成 `pouch full cell` 这样的新标签。全电池要求另用 cell_configuration 检查条件
   和明确的正负极对验证；外形名称不能代替电极与全电池要求。
   它描述的是 evidence composition obligation，不要求所有
   支持证据拥有相同 `experiment_id`。`constraint_ids` 只能引用本次 `constraint_records`；
   `context_fields` 只列对联合决策真正相关的字段；`target_context` 只填写用户明确给出的
   目标条件，未知字段保持未知，禁止为了让 join 成立而补默认值。例如：
   ```json
   {
     "joint_context_requirements": [
       {
         "group_id": "low-temp-high-voltage",
         "constraint_ids": ["retention", "voltage"],
         "context_fields": ["temperature_c", "electrode"],
         "target_context": {"temperature_c": -20.0, "electrode": "hard carbon"},
         "can_replan": true,
         "search_hint": "find evidence satisfying both hard constraints at the target operating context"
       }
     ]
   }
   ```
   只有约束确实需要共同实例化时才声明 joint group；互不相关的硬约束不要强行合组。
   原文条件检查器分别识别 `electrode` 与 `cell_configuration`。涉及电池形式的用户要求
   使用独立的 cell_configuration 条件；只有相关观测明确提供该字段时，才作联合检查。

## 候选与结果交接

候选由 Agent 阅读原文后整理，可直接传给 `screen_formulations`；无需先调用候选生成工具。
`query_formulation_space` 默认 `result_mode="evidence"`，只交回带出处的原文。
仅历史调用方需要规则候选草稿时选择 `result_mode="legacy_candidates"`；它不会代替语义核验。

对 named-study / named-candidate，要在声明候选归属或送入结构化筛选前核实组成来源：
至少一条 `evidence_kind="formulation"` 记录支持该候选。只有性能、温度或机理不能建立组成归属；
不得构造 `formulation_evidence_ids=[]` 并声称配方已由原文确认。可补取原文，或保留缺口结束。
可用 `search_paper_chunks(paper_id=..., query=..., top_k=8)` 追查；不要求补检索到固定数量。

筛选保留原样返回的 `screening_results`、`criteria_evaluations` 与 gaps，
不得手写分数、validations 或 scores。完整请求中的用户约束和候选判定不得为避错而删减。
若已声明 joint-context 合同，首次筛选和证据变化后的重新筛选都必须携带同一份合同；
只有用户改变目标才改变其要求。倍率和循环次数分开，多盐分别保存浓度，比例保留 mass/volume/molar/unknown。

`optimize_formulations` 只翻译已有 `DecisionState`；不得同时传入 `validations` 或 `scores`。
`hold` / `retrieve_more_evidence` 是当前评估反馈，Agent 可继续核验、换候选或解释限制结束。

## 显式请求旧版报告时

普通回答不需要 `synthesize_design_report`。兼容调用示例：
`{"request":{"task_type":"design","question":用户问题,"evidence_chunks": evidence_chunks,"candidates": candidates,"screening_results": screening_results}}`。
`claims` 只提交真实原文支持的事实和逐字引文；不为凑数量检索或生成断言。
`finalization_mode="limited"` 仅是这个旧接口的可选参数，不是 Agent 结束任务的前置步骤。
编译器若拒绝某项推荐，把缺口交回 Agent；不得把接口失败改写成实验性能失败。
报告正文、artifact、来源与实际 tool_call_id 按既有提交协议一致归属。
