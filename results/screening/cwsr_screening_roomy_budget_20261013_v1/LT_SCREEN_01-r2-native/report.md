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
- When the temperature dropped from 25 to -40 °C, the ionic conductivity of MP-based electrolytes only had a reduction of about 50%, yet PC-based electrolytes only remained 10%. [0038_Liu 等 - 2024 - Ester-based anti-freezing electrolyte achieving ultra-low temperature cycling for sodium-ion batteri::sec10::child4]（绑定级别：quote_bound）
  - 原文：When the temperature dropped from 25 to -40 °C, the ionic conductivity of MP-based electrolytes only had a reduction of about 50%, yet PC-based electrolytes only remained 10%.
- The low viscosity enables the electrolyte to maintain a high ionic conductivity at low temperatures. [0038_Liu 等 - 2024 - Ester-based anti-freezing electrolyte achieving ultra-low temperature cycling for sodium-ion batteri::sec10::child4]（绑定级别：quote_bound）
  - 原文：The low viscosity enables the electrolyte to maintain a high ionic conductivity at low temperatures.
- cyclic PC and EC are of high dielectric constant and high viscosity, while linear EMC, DMC and DEC feature are of much lower viscosity and relatively lower dielectric constant [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child0]（绑定级别：quote_bound）
  - 原文：cyclic PC and EC are of high dielectric constant and high viscosity, while linear EMC, DMC and DEC feature are of much lower viscosity and relatively lower dielectric constant
- increasing the $NaPF_{6}$ concentration in ethylene carbonate: dimethyl carbonate (EC:DMC 1:1 wt%) from 0.3 M to 1.4 M lowers the crystallisation temperature [0100_0046_Properties_of_NaPFsub6sub_electroly::sec1::child3]（绑定级别：quote_bound）
  - 原文：increasing the $NaPF_{6}$ concentration in ethylene carbonate: dimethyl carbonate (EC:DMC 1:1 wt%) from 0.3 M to 1.4 M lowers the crystallisation temperature
- the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity [0009_Cui 等 - 2025 - A Temperature-Adapted Ultraweakly Solvating Electrolyte for Cold-Resistant Sodium-Ion Batteries::sec0::child1]（绑定级别：quote_bound）
  - 原文：the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity
- The prepared HC anode exhibited a capacity of about 260 mAh g $^{-1}$ with a capacity retention of 87 % at -20 °C. [0055_Qiu 等 - 2024 - Research on low-temperature sodium-ion batteries Challenges strategies and prospect::sec9::child5]（绑定级别：quote_bound）
  - 原文：The prepared HC anode exhibited a capacity of about 260 mAh g $^{-1}$ with a capacity retention of 87 % at -20 °C.
- reversible capacity loss (1–3 months) dominated by electrolyte depletion and charge transfer polarization [0063_Tang 等 - 2025 - Quantitative decoupling of electrode expansion electrolyte depletion and dead sodium in low-temper::sec2::child0]（绑定级别：quote_bound）
  - 原文：reversible capacity loss (1–3 months) dominated by electrolyte depletion and charge transfer polarization
- The PC/EMC electrolyte cell displays the best cycling stability [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child2]（绑定级别：quote_bound）
  - 原文：The PC/EMC electrolyte cell displays the best cycling stability

### 后续实验建议（尚未执行）
- 在 −20 / −10 / 0 °C 三点实测各候选电解液的离子电导率（阻塞电极电导池或 EIS），核验模型排序，重点是 C1 与 C5 的低温名次以及 C3/C4 的近似并列。
- 用 DSC 测定各候选的结晶温度 Tc 与 NaPF6 盐析温度，确认 −20 °C 下仍保持液态且无盐析出（高 PC、低 EMC 组成如 C3、C4 尤其需要）。
- 对 C3 与 C4 建议并行测试：两者低温最低预测电导率几乎相同（1.242 vs 1.242 mS/cm），但 20 °C 预测差距明显（6.383 vs 5.904 mS/cm），EMC 近零的 C4 还需单独核验成膜能力与氧化稳定性。
- 在 HC || Na 半电池或 NFM/HC 软包电池中做 −10 °C 循环与 −20/−30 °C 倍率/容量保持率测试，并检查低温析钠与界面电阻。
- 评估低温长时存储下的电解液消耗（电解液耗尽与电荷转移极化）以及候选的成本与闪点，这些目标在本次初筛中均未量化。