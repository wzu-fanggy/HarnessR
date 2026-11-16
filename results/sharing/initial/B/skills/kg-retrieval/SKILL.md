---
name: kg_retrieval
description: 将用户自然语言查询转换为结构化知识图谱检索计划，确保高召回率和高精度
triggers: ["知识图谱", "图谱检索", "结构化查询", "KG 查询", "查询规划"]
---

# 知识图谱结构化检索

## 适用场景
用户需要查询钠离子电池电解液知识图谱中的实体关系（盐、溶剂、添加剂、配方、性能、机理、界面层等），且你希望召回率和召回质量最大化。

## 核心原则
- **结构化关系查询调用 `kg_query`，文本证据检索调用 `search_literature_chunks`**，两者可组合使用
- **问题涉及性能对比、筛选或用量/数值**（电导率、容量、循环次数、倍率、库仑效率、粘度、wt% 等）时，检索后**必须**调用 `extract_numeric_properties(evidence_chunks=<检索返回的 chunk 列表>)` 抽取数值证据，再组织回答
- **不要自行编造实体名**，必须使用规范名（canonical name）
- **实体名由工具内置解析器自动对齐 KG**：规范缩写（NaPF6）、全名（sodium hexafluorophosphate）、常见别名、diglyme 等都可直接使用，无需手动转换；解析失败时工具会原样透传
- **温度/条件关键词必须推断并填入查询参数**，不能留空
- **意图判断必须准确**，错误的 `query_type` 会导致返回结构不符合预期

## 实体规范化表（必须严格遵守）

### 盐类 (salts)
| 规范名 | 常见别名 |
|--------|----------|
| NaPF6 | sodium hexafluorophosphate, 六氟磷酸钠, NaPF₆ |
| NaFSI | sodium bis(fluorosulfonyl)imide, 双(氟磺酰)亚胺钠 |
| NaTFSI | sodium bis(trifluoromethanesulfonyl)imide |
| NaClO4 | sodium perchlorate, 高氯酸钠 |
| NaDFP | sodium difluoro(phosphato)borate |

### 溶剂 (solvents)
| 规范名 | 常见别名 |
|--------|----------|
| PC | propylene carbonate, 碳酸丙烯酯 |
| EC | ethylene carbonate, 碳酸乙烯酯 |
| EMC | ethyl methyl carbonate, 碳酸甲乙酯 |
| DEC | diethyl carbonate, 碳酸二乙酯 |
| DMC | dimethyl carbonate, 碳酸二甲酯 |
| DME | dimethoxyethane, 乙二醇二甲醚, dimethyl ether |
| DEGDME | diethylene glycol dimethyl ether |
| SL | sulfolane, 环丁砜 |

### 添加剂 (additives)
| 规范名 | 常见别名 |
|--------|----------|
| FEC | fluoroethylene carbonate, 氟代碳酸乙烯酯 |
| VC | vinylene carbonate, 碳酸亚乙烯酯 |
| PS | propylene sulfite |
| DTD | ethylene sulfate |
| PFPN | pentafluorophenyl nitrile |
| TMSP | tris(trimethylsilyl) phosphite |

### 电极材料 (electrodes)
| 规范名 | 常见别名 |
|--------|----------|
| Hard Carbon | hard carbon, HC, 硬碳 |
| NFM | NaFeMnO2, NaFe0.5Mn0.5O2 |

### 界面层 (interphases)
| 规范名 | 常见别名 |
|--------|----------|
| SEI | solid electrolyte interphase |
| CEI | cathode electrolyte interphase |

## 温度关键词推断规则

用户 query 中出现以下关键词时，**必须**把对应条件词用于检索参数：

| 用户表述 | 应填值 |
|----------|--------|
| 低温, low temperature, -40°C, -40℃ | `low temperature` |
| 零下, subzero, below 0, under 0 | `subzero` |
| 超低温, ultra-low temperature, -60°C | `ultra_low_temperature` |
| 室温, room temperature, 25°C | `room temperature` |
| 高温, high temperature, 60°C | `high temperature` |

> 注意：只填关键词短语，不要填完整句子。

## 意图到工具的映射规则

| 用户意图 | 工具调用 |
|----------|----------|
| "哪个更好" / "对比" / "vs" / "比较" | `kg_query(query_type="compare_materials", params={"material_a": "...", "material_b": "..."})` |
| "为什么" / "机理" / "形成原因" / "促进机制" | `search_literature_chunks(query="<英文机理查询>")`；查"什么能改善 X"时用 `kg_query(query_type="what_improves", params={"target": "..."})` |
| "推荐" / "筛选" / "适合" / "哪些候选" | `kg_query(query_type="find_formulations", params={...})`；带温度条件时用 `kg_query(query_type="find_materials_for_condition", params={"condition": "..."})` |
| 查某材料的性能/属性 | `kg_query(query_type="properties_of", params={"material": "..."})` |
| 查某盐可溶于哪些溶剂 | `kg_query(query_type="find_co_solvents", params={"salt": "..."})` |
| 其他（查配方组成、性能、关系） | `search_literature_chunks(query="<英文查询>")` |

`kg_query` 的 `params` 均支持可选 `"limit"`（默认 10）。

## 查询构造原则（先于一切策略）

- **默认用短主题查询**：3–6 个内容词，如 `NaPF6 NaFSI comparison carbonate electrolyte`。离线实测：短主题查询下目标文献稳定进入 top-5；把温度、单位、章节、实体过滤全部堆上去的长查询反而把最相关的论文挤出 top-10（v16 样本 2 实测踩中）。
- **先宽后窄**：第一轮查询不带 `sections`/`entities` 过滤；只有当首轮结果明显偏题、需要收窄时才加过滤条件。
- **结果偏题时把查询改短**，而不是加更多限定词。
- 对比/机理/综述类问题**不要**堆数据模式词；下面的"数据点检索策略"只适用于问题明确要求具体数值的场景。

## 数据点检索策略（性能数值类问题必用）

问题需要**具体性能数值**（电导率、容量、容量保持率、循环次数、倍率、库仑效率、粘度、用量 wt% 等）时，朴素语义查询容易只召回综述性铺垫而漏掉含数据的 chunk。必须组合以下三招：

- **query 里带数据模式词**：含数值论断的 chunk 通常同时带单位/数据词，query 中加入 `mAh g`、`mS cm`、`cycles`、`capacity retention`、`wt%`、`°C` 等能显著提升数据 chunk 命中
- **用 `sections` 限定章节**：`sections=["results"]` 聚焦 Results/Discussion（过滤是大小写不敏感的子串匹配，`"results"`/`"experimental"`/`"conclusion"` 均可），避免引言和综述段落稀释 top_k
- **跨论文对比数据时用 `search_literature_multiview`**：把同一数据点拆成 2–3 个角度查询（如材料角度+性能角度+条件角度），RRF 融合后排在前面
- 检索到 chunk 后仍须调 `extract_numeric_properties` 结构化数值，再组织回答

### 参数论断主动验证

对于要作为事实写出的具体数值论断，在组织答案前调用 `verify_parameter_claim(claim="...")`：

- 它会对论断中的化学实体和数值生成 narrow query，验证该数值是否出现在召回 chunk 中；
- 验证通过后再引用，避免把宽泛检索未覆盖到的物性表数值误写入答案；
- 验证失败时说明"证据未覆盖"或改用定性表述。

## 设计类任务检索：组分 + 机理双轨

设计类任务若只用“组分罗列式”查询（如 `NaPF6 EC PC DMC -20°C sodium ion battery`），召回的多为配方表、综述铺垫或通用描述，容易漏掉**机理论断**和**宽温域/性能目标导向**的关键 chunk（这些 chunk 常以 `wide temperature range`、`desolvation`、`interfacial stability`、`NaF-rich inorganic interface` 等为中心）。离线诊断显示：

- 组分导向查询下，某 -20°C 碳酸酯基电解液设计样本的 GT 锚定文献全部排在 top-50 之外，导致 `context_recall=0`；
- 改为机理-性能导向查询后，同一锚定文献（宽温域添加剂、低温去溶剂化、NaF 界面）分别进入 **top-3 / top-8 / top-8**。

因此，设计/逆向设计类问题在检索时必须**双轨并行**：

### 1. 组分导向轨
列出目标盐、溶剂、添加剂、温度条件，用于召回配方与组分对比证据。示例：
`NaPF6 EC PC DMC EMC FEC carbonate electrolyte -20°C sodium ion hard carbon`

### 2. 机理-性能导向轨
用 1–3 个以“性能目标 + 机理/界面关键词”为核心的英文查询：

| 目标 | 推荐查询关键词 | 目的 |
|---|---|---|
| 宽温域 | `dual-functional additive wide temperature range sodium-ion battery NFM hard carbon` | 召回含 NaDFP 等添加剂的全温域性能 chunk |
| 低温 | `low temperature desolvation sluggish sodium ion transport carbonate electrolyte` | 召回低温限制机理 chunk |
| 界面 | `fluorine-rich additive NaF inorganic interface stability sodium battery` | 召回含氟添加剂形成 NaF 界面的 chunk |
| 添加剂机理 | `additive mechanism interphase SEI CEI low temperature sodium-ion` | 召回成膜/界面调控机理 chunk |

### 执行建议
- 先调用 `search_literature_chunks` / `search_literature_multiview` 获取组分轨证据；
- **必须**再调用一次 `search_literature_chunks` / `search_literature_multiview` 获取机理-性能轨证据；
- 需要具体数值时，机理-性能查询中加入数据模式词（`capacity retention`、`cycles`、`wt%`、`°C` 等），并限定 `sections=["results"]`；
- 两轨证据合并后再进入候选生成/筛选/评分。

## Few-shot 示例

### 示例 1：多实体 + 温度筛选
**用户**：低温下 NaPF6 和 PC 配方的电解液循环性能怎么样？

**调用**：
```json
{
  "tool": "kg_query",
  "args": {
    "query_type": "find_formulations",
    "params": {"salt": "NaPF6", "solvent": "PC", "limit": 10}
  }
}
```
随后用 `search_literature_chunks(query="NaPF6 PC electrolyte low temperature cycling performance")` 补充文献证据。

### 示例 2：对比两种盐
**用户**：NaPF6 和 NaFSI 在碳酸丙烯酯里的性能哪个好？

**调用**：
```json
{
  "tool": "kg_query",
  "args": {
    "query_type": "compare_materials",
    "params": {"material_a": "NaPF6", "material_b": "NaFSI", "limit": 10}
  }
}
```

### 示例 3：界面层机理
**用户**：FEC 为什么能促进 SEI 形成？

**调用**：
```json
{
  "tool": "search_literature_chunks",
  "args": {"query": "FEC promotes SEI formation mechanism sodium-ion battery"}
}
```

### 示例 4：纯配方查找
**用户**：含 FEC 和 VC 的电解液配方有哪些？

**调用**：
```json
{
  "tool": "kg_query",
  "args": {
    "query_type": "find_formulations",
    "params": {"additive": "FEC", "limit": 10}
  }
}
```
再对 VC 重复一次，或改用 `search_literature_chunks(query="electrolyte formulations containing FEC and VC")`。

### 示例 5：宽温域 + 性能
**用户**：硬碳负极在零下温度长循环的电解液推荐

**调用**：
```json
{
  "tool": "kg_query",
  "args": {
    "query_type": "find_materials_for_condition",
    "params": {"condition": "subzero", "limit": 10}
  }
}
```
随后用 `search_literature_chunks(query="hard carbon anode subzero temperature long cycle life electrolyte")` 补充证据。

### 示例 6：具体性能数据点检索
**用户**：DME 基电解液在 -20°C 下硬碳的容量和循环保持率是多少？

**调用**：
```json
{
  "tool": "search_literature_chunks",
  "args": {
    "query": "DME ether electrolyte hard carbon -20°C mAh g capacity retention cycles",
    "top_k": 8,
    "sections": ["results"]
  }
}
```
随后必须调 `extract_numeric_properties(evidence_chunks=<返回的 chunk 列表>)` 抽取结构化数值。若结果不足，再用 `search_literature_multiview(queries=["DME electrolyte low temperature capacity mAh g", "glyme chain length hard carbon -20°C cycles retention"])` 扩展。

## 常见错误（严禁）

- ❌ **使用中文或全称填入实体参数**：如 `"六氟磷酸钠"`、`"sodium hexafluorophosphate"`，必须用 `"NaPF6"`
- ❌ **condition 填完整句子**：如 `"在低温环境下测试"`，应只填 `"low temperature"`
- ❌ **漏掉条件参数**：用户明确提到温度、循环、倍率时，不能忽略
- ❌ **compare_materials 只填一个材料**：对比必须提供 `material_a` 和 `material_b` 两个对象
- ❌ **机理查询只用 kg_query**：KG 结构化关系不含详细机理描述，机理类问题必须配合 `search_literature_chunks` 获取文本证据
- ❌ **query 参数填完整句子**：如 `"我想知道循环性能怎么样"`，应只填 `"cycling performance"` 等关键词短语
- ❌ **数值类问题只用朴素语义查询**：不带数据模式词（mAh g、mS cm、cycles）、不限 `sections=["results"]`，含数据的 chunk 会被综述性段落挤出 top_k

## 工具调用方式

1. 分析用户意图，提取关键实体（使用规范名）和条件
2. 按"意图到工具的映射规则"选择 `kg_query` 的 `query_type` 并构造 `params`，或构造简洁的英文查询语句
3. 结构化关系调用 **`kg_query`**，文本证据调用 **`search_literature_chunks`**
4. 需要从多个角度扩展时，可再用 `search_literature_multiview`（多查询融合）或 `explore_entity_neighborhood`（实体邻居子图）
5. 证据涉及性能数值、用量或对比数据时，调用 `extract_numeric_properties` 抽取结构化数值后再回答

## 引用纪律

- 只能引用检索实际返回的文献（以 chunk 的 `paper_title` 为准）；综述中转述的原始文献结论，引用时落到实际读到的综述并标注"转引自 X"，禁止伪装成已获取一手文献；具体规则见 `electrolyte_design` skill 的"引用纪律"。
