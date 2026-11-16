## 电解液配方筛选

### 候选组成
- C1：全组分摩尔比例为 NaPF6=0.128283，EC=0.569494，EMC=0.284961，PC=0.0172623
- C2：全组分摩尔比例为 NaPF6=0.199518，EC=0.175088，EMC=0.361687，PC=0.263708
- C3：全组分摩尔比例为 NaPF6=0.136592，EC=0.0991394，EMC=0.221273，PC=0.542996
- C4：全组分摩尔比例为 NaPF6=0.0976679，EC=0.0135361，EMC=0.00117612，PC=0.88762
- C5：全组分摩尔比例为 NaPF6=0.0852192，EC=0.173613，EMC=0.736773，PC=0.00439533

### 电导率预测
模型：na-conductivity-rf-v1。数值单位为 mS/cm。

| 候选 | -20 °C | -10 °C | 0 °C | 20 °C |
| --- | --- | --- | --- | --- |
| C1 | 1.4178 | 2.3060 | 3.4065 | 7.4089 |
| C2 | 0.4988 | 1.1785 | 2.0497 | 4.6263 |
| C3 | 1.2419 | 2.1329 | 3.2608 | 6.3833 |
| C4 | 1.2416 | 1.8005 | 3.1849 | 5.9043 |
| C5 | 2.3004 | 3.1813 | 3.8823 | 5.9742 |

### 初筛结果
在 -20、-10、0 °C 的 na-conductivity-rf-v1 模型预测中，各候选最低电导率为 C1=1.41785、C2=0.498752、C3=1.24186、C4=1.2416、C5=2.30043 mS/cm。按最低预测电导率最大这一初筛指标，C5的数值最高，可优先开展低温输运实验验证。这一排序来自模型计算，不等于电池容量或循环性能的实验排名。在参照温度 20 °C，预测值为 C1=7.40887、C2=4.62632、C3=6.3833、C4=5.90434、C5=5.97425 mS/cm，数值最高的候选为 C1。

### 文献依据
## 声明绑定状态（语义蕴含仍未自动验证）
- In contrast, the ionic conductivity of PC/EMC electrolyte decreases much slowly than other electrolytes and no crystallization occurs even at -40 °C (Fig.2c). [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child3]（绑定级别：quote_bound）
  - 原文：In contrast, the ionic conductivity of PC/EMC electrolyte decreases much slowly than other electrolytes and no crystallization occurs even at -40 °C (Fig.2c).
- While the cells with EC based electrolytes almost do not operate below -20 °C (Fig. 3c). [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child5]（绑定级别：quote_bound）
  - 原文：While the cells with EC based electrolytes almost do not operate below -20 °C (Fig. 3c).
- At low temperatures, the pouch cell with PC/EMC electrolyte shows higher capacity retention, i.e., 81.9%, 72.7% and 45.8% of the capacity at room temperature at -20 °C, -30 °C, and -40 °C, respectively. [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child4]（绑定级别：quote_bound）
  - 原文：At low temperatures, the pouch cell with PC/EMC electrolyte shows higher capacity retention, i.e., 81.9%, 72.7% and 45.8% of the capacity at room temperature at -20 °C, -30 °C, and -40 °C, respectively.
- tend to experience a sharp decline in conductivity at low temperatures. [0086_Zhou 等 - 2025 - A Nonfluorinated Diluent in Ether-Based Electrolyte for Low-Temperature Sodium-Ion Batteries::sec1::child3]（绑定级别：quote_bound）
  - 原文：tend to experience a sharp decline in conductivity at low temperatures.
- However, the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity. [0009_Cui 等 - 2025 - A Temperature-Adapted Ultraweakly Solvating Electrolyte for Cold-Resistant Sodium-Ion Batteries::sec0::child1]（绑定级别：quote_bound）
  - 原文：However, the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity.
- DMC as co-solvent to decrease viscosity and improve ionic conductivity [0150_aenm202000093pdf::sec4::child9]（绑定级别：quote_bound）
  - 原文：DMC as co-solvent to decrease viscosity and improve ionic conductivity
- Because of their excellent low-temperature (-15 to -40 °C) tolerance, sodium-ion batteries are emerging as a complement to lithium-ion batteries for use in extremely cold environments [0008_Cui 等 - 2025 - A review of ways to improve the performance of hard carbon anodes in low-temperature sodium-ion batt::sec1::child0]（绑定级别：quote_bound）
  - 原文：Because of their excellent low-temperature (-15 to -40 °C) tolerance, sodium-ion batteries are emerging as a complement to lithium-ion batteries for use in extremely cold environments

### 后续实验建议（尚未执行）
- 对 C5（NaPF6 0.0852 / EC 0.1736 / EMC 0.7368 / PC 0.0044）与次优的 C1 在 −20、−10、0、20 °C 各温度恒温平衡后用同一电导池与同一批盐/溶剂独立复测离子电导率（EIS），每个温度至少三次平行，以检验模型排序与 ±1σ 不确定度（C5 在 −20 °C 约 1.23 倍因子）是否重叠。
- 对 C3 与 C4（−20 °C 预测 1.242 与 1.242 mS cm-1，差值远小于模型不确定度）并行开展电导率复测，不依据模型差异淘汰其中任一配方。
- 在统一电极、电芯形式、倍率与电压窗口下比较优先候选：例如以硬碳（HC）为负极、NaNi1/3Fe1/3Mn1/3O2 或 Na3V2(PO4)2F3 为正极组装扣式或软包电芯，在 −20、−10、0、20 °C 以相同倍率测容量保持、直流内阻与阻抗谱，验证电导率优势是否转化为低温容量与倍率表现。
- 补充低温相行为与安全相关表征：对 C4（近纯 PC）、C3（PC 主导）测 DSC 结晶/凝固温度与低温黏度，确认模型未覆盖的冻结风险；对 C5（EMC 富集、低盐）关注低盐浓度下的离子对与钠沉积风险。
- 若复测显示低温电导排序与模型不一致，则按实测值重新排序并仅保留实测最低电导更高的 1–2 个配方进入电芯级验证。