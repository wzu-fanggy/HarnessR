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
- they determine the concentration of free ions and their mobility, thereby regulating electrolyte conductivity [0054_Qin 等 - 2024 - Carboxylate ester-based electrolytes for Na-ion batteries::sec2::child2]（绑定级别：quote_bound）
  - 原文：they determine the concentration of free ions and their mobility, thereby regulating electrolyte conductivity
- Classical carbonates, such as ethylene carbonate/propylene carbonate (EC/PC)-based electrolytes with high viscosity and strong solvation ability to $Na^{+}$ , tend to experience a sharp decline in conductivity at low temperatures. [0086_Zhou 等 - 2025 - A Nonfluorinated Diluent in Ether-Based Electrolyte for Low-Temperature Sodium-Ion Batteries::sec1::child3]（绑定级别：quote_bound）
  - 原文：Classical carbonates, such as ethylene carbonate/propylene carbonate (EC/PC)-based electrolytes with high viscosity and strong solvation ability to $Na^{+}$ , tend to experience a sharp decline in conductivity at low temperatures.
- EC has the highest dielectric constant (89.78), indicating its ability to dissolve enough salts. [0305_inf212023pdf::sec8::child0]（绑定级别：quote_bound）
  - 原文：EC has the highest dielectric constant (89.78), indicating its ability to dissolve enough salts.
- finding that 1 M NaPF $_{6}$ offers the highest bulk conductivity [0100_0046_Properties_of_NaPFsub6sub_electroly::sec1::child0]（绑定级别：quote_bound）
  - 原文：finding that 1 M NaPF $_{6}$ offers the highest bulk conductivity
- DMC as co-solvent to decrease viscosity and improve ionic conductivity [0150_aenm202000093pdf::sec4::child9]（绑定级别：quote_bound）
  - 原文：DMC as co-solvent to decrease viscosity and improve ionic conductivity

### 后续实验建议（尚未执行）
- 对 C5（并行的 C1、C3、C4）在实际实验室条件下用同一电导率池/对称电池 + EIS 独立复测 −20、−10、0 与 20 °C 的电导率，核对模型预测（C5：2.30 / 3.18 / 3.88 / 5.97 mS/cm）；同时测定黏度、熔点/结晶温度，确认 −20 °C 以下无 NaPF6 析出或溶剂冻结。
- 在相同电极、相同电芯类型、相同倍率与电压范围下做电池级比较：例如硬碳 || 钠半电池（0.001–2 V，0.1C 充放）与硬碳 || 正极全电池，在 −20 °C 与 20 °C 分别测容量保持、循环稳定性与 EIS 界面阻抗，用以检验“电导率最高的 C5 是否同时具有最低界面阻抗与最优低温容量保持”。
- 对 C3 与 C4 的并列预测（−20 °C 均为 1.24 mS/cm，差异远小于模型 1σ 因子 1.20–1.26）做重点复核：C4 组成几乎为纯 PC（EC 0.0135、EMC 0.0012），其模型特征接近定义边界，预测外推风险较高，需实测验证而非按排序排除。
- 若上述电导率复测的排序发生改变（尤其 C1 与 C5 之间），以复测值重排优先级；本报告的排序仅基于 na-conductivity-rf-v1 在三点温度上的最小值，不含任何通过阈值。