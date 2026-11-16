# 配方计量求解：数学输入与输出

入口：经run_script运行 electrolyte_design/scripts/solve_formulation.py。
参数二选一：--request-json <JSON对象>，或 --request-file <Orion工作区JSON路径>。
新接口不需要原文文件、来源ID或文字假设。Agent负责理解原文、选定解释，将解释转换为明确约束；原文与解释保留在任务记录中。每次调用描述一个混合状态，不能把添加剂加入前后的关系当成同一状态。

## 输入：只含三个字段

- components：明确物种ID列表；从同目录formulation_catalog.json读取物种及物性。当前含五个已取得PubChem记录。不要把缩写直接当ID，不自填摩尔质量、不猜未知物种。
- constraints：计量约束列表，字段见下表。可给非空字符串id；省略时依次使用constraint_1、constraint_2等。
- normalization：{"components":["ID",...],"amount":1,"unit":"mol"}，或单位g。这是数学归一化尺度，不是实测制备批量。

| type | 字段与定义 |
|---|---|
| mole_ratio / mass_ratio | components：相关物种ID；values：对应正比例数组。按所选类型解释摩尔/质量比 |
| mole_fraction / mass_fraction | component：分子物种ID；basis：明确分母，须含分子；value；unit：fraction或percent |
| molarity | component；value；unit：mol/L；volume_basis：final_solution |
| density | value；unit：g/mL；volume_basis：final_solution。指整个溶液密度 |
| moles / mass | component；value；unit分别为mol / g |

分数区间可用lower与upper替代value，不能同时传名义值和区间。不得把分析者敏感性区间称为原文误差。
比例基准必须明确；体积比不在当前支持范围，不能改标签当摩尔比。溶剂体积不等于最终溶液体积。
归一化与绝对用量约束都参加计算；若用mass/moles输入绝对量，须确保归一化与实际批量相容，不随意另设1 mol归一化。

完整请求示例：EC、PC摩尔比1:1，溶剂合计按1 mol归一化。下面两个ID分别是EC和PC，不是名称映射接口。

```json
{"components":["pubchem:7303","pubchem:7924"],"constraints":[{"type":"mole_ratio","components":["pubchem:7303","pubchem:7924"],"values":[1,1]}],"normalization":{"components":["pubchem:7303","pubchem:7924"],"amount":1,"unit":"mol"}}
```

## 输出：四种status

- determined：返回amounts_mol与masses_g，键为原组分ID。示例中分别为0.5/0.5 mol、44.03/51.045 g。
- underdetermined：返回amount_ranges_mol与fraction_ranges，不返回虚假的完整点组成。上下界相等的组分已可确定；上界未知时有upper_unbounded=true。
- infeasible：返回conflicting_constraints。它们在给定摩尔质量、归一化及非负条件下冲突，不直接认定原论文错误。
- invalid_input：返回reason及recoverable=true，补正输入后可重试。

可求解结果同时保留输入归一化和各组分数量/分数范围。涉及浓度/密度时，另返回solution_volume_range_L；唯一解可含solution_volume_L与implied_density_g_mL。未输入密度时反推值仍是数学结果，不是实测。
界限可能为极限，fraction_ranges内保留相应说明；未知上界不解释为实际可配制性。
CLI输入错误退出码2；可确定、欠定和冲突都是有效计算结果，退出码0。run_script非零时仍可读取stdout中的JSON诊断。
结果仅适用于输入的计量关系，不判断原文语义、溶解度、液态、电化学稳定性或电池性能。含添加剂的组成不能直接送入不支持添加剂的性质模型。

## 旧接口兼容

已保存的component_ids/state_id/assumptions/claims/normalization请求仍可显式传--evidence-file，保留原返回字段和状态；新任务使用上面的三字段合同。
新数学接口不接受--evidence-file，避免混用两套合同。旧接口的引用存在检查也不等于语义核验，不用于新接口准入。
