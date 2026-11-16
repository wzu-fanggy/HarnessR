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
- typically made of 1 M $NaClO_{4}$ or $NaPF_{6}$ dissolved into a blend of cyclic carbonate and linear carbonate (Table S2†), generally has the highest conductivity [0054_Qin 等 - 2024 - Carboxylate ester-based electrolytes for Na-ion batteries::sec2::child1]（绑定级别：quote_bound）
  - 原文：typically made of 1 M $NaClO_{4}$ or $NaPF_{6}$ dissolved into a blend of cyclic carbonate and linear carbonate (Table S2†), generally has the highest conductivity
- the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity [0009_Cui 等 - 2025 - A Temperature-Adapted Ultraweakly Solvating Electrolyte for Cold-Resistant Sodium-Ion Batteries::sec0::child1]（绑定级别：quote_bound）
  - 原文：the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity
- The good power capability of these two cells can be attributed to high ionic conductivity of the electrolytes, which results in lower charging and discharge over-potentials [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child3]（绑定级别：quote_bound）
  - 原文：The good power capability of these two cells can be attributed to high ionic conductivity of the electrolytes, which results in lower charging and discharge over-potentials

### 后续实验建议（尚未执行）
- 对 C5（初筛指标 2.30 mS/cm）与 C1（1.42 mS/cm，且模型 1σ 因子约 1.55）在 −20、−10、0、20 °C 用同一批 NaPF6 与溶剂、同一手套箱水氧条件下独立复测离子电导率（推荐 EIS/交流阻抗法），核验模型排序，尤其是 C1 与 C5 在 −20 °C 的差距是否真实存在。
- C3 与 C4 的 −20 °C 预测几乎持平（1.242 与 1.242 mS/cm，差异 <0.1%，模型 1σ 因子分别约 1.26 与 1.20），建议并行测试：同时测 −20 °C 电导率、黏度与 DSC 凝固/结晶行为，评估 PC 高含量配方（C4 的 PC 摩尔份 0.888、EMC 仅 0.0012）是否存在盐析出或溶剂结晶风险。
- 在相同电极、相同电芯构型（例如 HC||Na 半电池与 HC||Na3V2(PO4)2F3 全电池各一组）、相同倍率与相同电压窗口下，对优先候选做 −20 °C 的倍率性能、恒流循环与 EIS 阻抗对比，报告相对室温的容量保持率与首效；这些电池级指标尚未在本次工作中获得。
- 由于 20 °C 电导率排序（C1 7.41 > C3 6.38 > C5 5.97 > C4 5.90 > C2 4.63 mS/cm）与低温初筛排序不一致，若应用同时要求室温功率与低温可用性，应对 C1 与 C5 并行开展上述复测与电池验证，而不是仅按室温数值下结论。