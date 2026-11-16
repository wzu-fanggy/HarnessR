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
在 -20、-10、0 °C 的 na-conductivity-rf-v1 模型预测中，各候选最低电导率为 C1=1.41785、C2=0.498752、C3=1.24186、C4=1.2416、C5=2.30043 mS/cm。按最低预测电导率最大这一初筛指标，优先开展后续实验的前2种候选依次为C5（2.3004 mS/cm）、C1（1.4178 mS/cm）。这一排序来自模型计算，不等于电池容量或循环性能的实验排名。在参照温度 20 °C，预测值为 C1=7.40887、C2=4.62632、C3=6.3833、C4=5.90434、C5=5.97425 mS/cm，数值最高的候选为 C1。

### 文献依据
## 声明绑定状态（语义蕴含仍未自动验证）
- electrolytes play a pivotal role in governing low-temperature performance for three reasons [0054_Qin 等 - 2024 - Carboxylate ester-based electrolytes for Na-ion batteries::sec2::child2]（绑定级别：quote_bound）
  - 原文：electrolytes play a pivotal role in governing low-temperature performance for three reasons
- the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity [0009_Cui 等 - 2025 - A Temperature-Adapted Ultraweakly Solvating Electrolyte for Cold-Resistant Sodium-Ion Batteries::sec0::child1]（绑定级别：quote_bound）
  - 原文：the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity
- fast ion transport process and charge transfer reactions are both great challenges for a cathode to work at low temperature [0012_Deng 等 - 2022 - Self-optimizing weak solvation effects achieving faster low-temperature charge transfer kinetics for::sec3::child1]（绑定级别：quote_bound）
  - 原文：fast ion transport process and charge transfer reactions are both great challenges for a cathode to work at low temperature

### 后续实验建议（尚未执行）
- 对 C5 与 C1 在 −20、−10、0、20 °C 用同一批新配电解液、同一电导率测试方法（同一电导池或 EIS 电池、同一恒温平衡时间）独立复测离子电导率，核对模型排序及 1σ 区间（C5 约 ×1.23，C1 约 ×1.55）。
- C3 与 C4 的 −20 °C 预测值几乎相同（1.2419 与 1.2416 mS/cm），且二者的 1σ 区间完全重叠，建议作为并列候选在同一批测试中并行开展，而不是仅凭模型值排序。
- 在相同电极（硬碳负极）、相同电池构型（HC||Na 半电池，或与同一正极匹配的 HC||NFM 全电池）、相同倍率（0.1–0.2 C）与相同电压窗口（半电池 0.01–2.0 V vs Na/Na+，全电池 2.0–4.2 V）下，比较 C5/C1 及 C3/C4 在 −20、−10、0 °C 的容量、极化与循环保持率；这些实验尚未开展。
- 补做 −20 °C 低温静置后的析盐/凝固观察与 20 °C 老化后电导率复测，重点核查高 EC 的 C1 与近纯 PC 的 C4，因文献报道 NaPF6 在 −20 °C 以下会析出而 PC 基体系低温电导率衰减更陡。
- 增补 EIS（电荷转移电阻与活化能）与界面（SEI/CEI）表征，用以判定电导率差异是否真正转化为低温界面动力学差异。
- 在 20 °C 参照温度上同步比较，验证室温电导率排序（C1 > C3 > C5 > C4 > C2）不能替代低温初筛排序，避免用室温数据替代低温选型。