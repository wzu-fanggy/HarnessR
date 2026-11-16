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
在 -20、-10、0 °C 的 na-conductivity-rf-v1 模型预测中，各候选最低电导率为 C1=1.41785、C2=0.498752、C3=1.24186、C4=1.2416、C5=2.30043 mS/cm。按最低预测电导率最大这一初筛指标，C5的数值最高，可优先开展低温输运实验验证。这一排序来自模型计算。在参照温度 20 °C，预测值为 C1=7.40887、C2=4.62632、C3=6.3833、C4=5.90434、C5=5.97425 mS/cm，数值最高的候选为 C1。

### 文献依据
## 声明绑定状态（语义蕴含仍未自动验证）
- due to sodium ion kinetics and thermodynamics limitations, sodium-ion batteries are still challenging to operate at high rates and extreme temperature conditions [0028_Li 等 - 2024 - Hierarchical doping electrolyte solvation engineering to achieve high-performance sodium-ion batteri::sec3::child0]（绑定级别：quote_bound）
  - 原文：due to sodium ion kinetics and thermodynamics limitations, sodium-ion batteries are still challenging to operate at high rates and extreme temperature conditions
- The electrolyte properties such as ionic conductivity and ion mobility at the electrode-electrolyte interfaces are primarily responsible for the performance decay of the battery at low temperature. [0073_Yang 等 - 2023 - Entropy-Driven Solvation toward Low-Temperature Sodium-Ion Batteries with Temperature-Adaptive Featu::sec0::child1]（绑定级别：quote_bound）
  - 原文：The electrolyte properties such as ionic conductivity and ion mobility at the electrode-electrolyte interfaces are primarily responsible for the performance decay of the battery at low temperature.
- the resistance of electrolyte increases rapidly as temperature drops because of high freezing points of most carbonate solvents [0073_Yang 等 - 2023 - Entropy-Driven Solvation toward Low-Temperature Sodium-Ion Batteries with Temperature-Adaptive Featu::sec0::child1]（绑定级别：quote_bound）
  - 原文：the resistance of electrolyte increases rapidly as temperature drops because of high freezing points of most carbonate solvents
- When the temperature is lower than -20 °C, the ionic conductivity of CCE decreases rapidly [0033_Li 等 - 2026 - Enhanced sodium storage in hard carbon via solvent co-intercalation electrolyte enabling Ah-level po::sec2::child6]（绑定级别：quote_bound）
  - 原文：When the temperature is lower than -20 °C, the ionic conductivity of CCE decreases rapidly
- sodium salts will also precipitate and partially solidify due to the poor solubility of Na salt under cold ambient conditions [0020_Feng 等 - 2024 - Monolithic Interphase Enables Fast Kinetics for High-Performance Sodium-Ion Batteries at Subzero Tem::sec2::child2]（绑定级别：quote_bound）
  - 原文：sodium salts will also precipitate and partially solidify due to the poor solubility of Na salt under cold ambient conditions
- Additionally, the thick electrode maintains a high capacity retention rate of 87% at -20 °C. [0008_Cui 等 - 2025 - A review of ways to improve the performance of hard carbon anodes in low-temperature sodium-ion batt::sec16::child4]（绑定级别：quote_bound）
  - 原文：Additionally, the thick electrode maintains a high capacity retention rate of 87% at -20 °C.

### 后续实验建议（尚未执行）
- 对 C5 与 C1 开展 −20/−10/0/20 °C 变温离子电导率实测（同一电导池、同一温度阶梯），核对模型预测值与 predict 报告的 1σ 区间（C5 约 ±23%、C1 约 ±55%–77%）是否重叠；本任务不预设通过阈值，比较应以区间重叠与否表述。
- 对 C2（盐摩尔分数最高）单独做低温析盐/凝固点观察（−20 °C 静置后目视与 DSC），核实模型给出的最低低温电导率是否源于盐析出风险；文献提示 NaPF6 在低温下可析出并使电导率骤降。
- 在 Na||HC 半电池上测试 C5 与 C1 的 −20 °C 放电容量占 20 °C 容量比例与 −20 °C 循环保持率，并同步记录电极负载量与电池构型；文献中的低温保持率均为特定电极/体系实测结果，不能直接迁移为本候选的电池性能。
- 补充 O3 层状氧化物 || 硬碳全电池在 −20/−10/0 °C 的循环与 −20 °C 长期搁置（含可逆/不可逆容量损失拆解）验证，因为低温性能衰减机制包含电解液消耗、界面恶化与死钠积累等多个阶段。
- 本次为模型性质初筛，未评估成本、闪点/安全性与界面成膜；若后续要求这些指标，需要另行检索或实验，不应沿用本轮预测结论。