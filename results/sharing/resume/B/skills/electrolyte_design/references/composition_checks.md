# 已有组成检查工具的范围

原生工具名保留`check_formulation_feasibility`，但它只检查组成输入的完整性/一致性并报告参考物性，不判断实际能否配制。返回`schema_version=2`。

## 输入

`request.candidates`为非空列表。每个候选给`salt`及组分绝对物质的量`component_mmol`（兼容`components_mmol`），数值非负且有限；盐和溶剂的数量须明确。
可附带`solvent_ratios`及`solvent_ratio_basis=molar/mass/volume`，或明确的`solvent_molar_shares`。使用既有组分注册表，不猜未知物种。

绝对用量与相对比例的合并规则：

- 只有盐1 mmol和溶剂摩尔比1:1，仍缺溶剂总量，返回`insufficient`。
- 若另知EC为4 mmol、EC:EMC摩尔比1:3，可补EMC为12 mmol。
- 完整绝对用量与附带比例冲突，返回`invalid`，不能忽略比例。
- 两个绝对用量别名或两个比例表示也必须一致；数值表示核对兼容生成器六位小数摘要，这不是科学判断阈值。
- 质量比按目录摩尔质量换算相对摩尔权重；体积比另用目录25°C纯组分密度近似换算。这些换算仍需要已知溶剂用量定标，不能给出最终溶液体积或mol/L。

`temperature_c`可选。`rules`可显式给工程参考阈值；默认不设凝固、黏度或最小用量准入阈值。盐摩尔分数训练范围仅用于参考报告，不代替预测器完整适用域检查。

## 输出

- `composition_status=complete`：当前组成输入完整且数值表示相容，可查看盐摩尔分数及其他明确量。
- `composition_status=insufficient`：缺量或比例基准，返回缺口，不补造完整配方。
- `composition_status=invalid`：数值非法、物种未知或表示冲突，返回原因。

`feasible`与汇总中的同名字段始终为`null`；`physical_feasibility=not_evaluated`。不要把null当False淘汰配方，也不能把composition_status=complete当实际可配制通过。

`model_assessable=false`仅说明已发现组成缺失或当前模型不支持等问题；`null`说明需调用实际预测器检查完整组成/温度域。此工具不加载RF或给出预测。
`not_evaluable`针对当前组成输入，不能写成配方性能差。

## 物性解释

黏度为25°C无盐混合溶剂的理想混合参考，不是目标温度下的含盐电解液黏度。
纯组分熔点和调用者给的占比规则只能提示，不触发规则也不表示混合物保持液态；混合凝固点未评价。
纯组分闪点作为参考列表返回，不输出混合物闪点下限或安全通过。当前工具不评价相图、溶解度、界面、循环或安全。

较复杂计量关系需要求解时，按需使用[配方计量数学接口](formulation_stoichiometry.md)。它的`infeasible`指输入数学约束冲突，与物理不可配制不是同一结论。
