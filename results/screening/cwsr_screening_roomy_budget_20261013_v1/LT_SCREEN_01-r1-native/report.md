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
- Electrolyte is a medium responsible for ionic transport [0184_jijhydene201512090pdf::sec5::child0]（绑定级别：quote_bound）
  - 原文：Electrolyte is a medium responsible for ionic transport
- Ionic conductivity is critical to evaluate the performance of electrolytes, especially at low temperatures [0038_Liu 等 - 2024 - Ester-based anti-freezing electrolyte achieving ultra-low temperature cycling for sodium-ion batteri::sec10::child4]（绑定级别：quote_bound）
  - 原文：Ionic conductivity is critical to evaluate the performance of electrolytes, especially at low temperatures
- The low viscosity enables the electrolyte to maintain a high ionic conductivity at low temperatures [0038_Liu 等 - 2024 - Ester-based anti-freezing electrolyte achieving ultra-low temperature cycling for sodium-ion batteri::sec10::child4]（绑定级别：quote_bound）
  - 原文：The low viscosity enables the electrolyte to maintain a high ionic conductivity at low temperatures
- the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity [0009_Cui 等 - 2025 - A Temperature-Adapted Ultraweakly Solvating Electrolyte for Cold-Resistant Sodium-Ion Batteries::sec0::child1]（绑定级别：quote_bound）
  - 原文：the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity
- sodium ion batteries still suffer from severe performance degradation at low temperatures due to the conflict between ion desolvation and diffusion [0033_Li 等 - 2026 - Enhanced sodium storage in hard carbon via solvent co-intercalation electrolyte enabling Ah-level po::sec1::child0]（绑定级别：quote_bound）
  - 原文：sodium ion batteries still suffer from severe performance degradation at low temperatures due to the conflict between ion desolvation and diffusion
- the performance of the battery will decline sharply due to the limitation of sodium ion kinetics and thermodynamics [0028_Li 等 - 2024 - Hierarchical doping electrolyte solvation engineering to achieve high-performance sodium-ion batteri::sec9::child0]（绑定级别：quote_bound）
  - 原文：the performance of the battery will decline sharply due to the limitation of sodium ion kinetics and thermodynamics
- DMC as co-solvent to decrease viscosity and improve ionic conductivity [0150_aenm202000093pdf::sec4::child9]（绑定级别：quote_bound）
  - 原文：DMC as co-solvent to decrease viscosity and improve ionic conductivity
- EC based electrolytes have a side reaction (bubble) between 4.0 V to 4.5 V and mingle decomposition, while PC based electrolytes can be stable until about 4.2 V then slightly decompose [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child4]（绑定级别：quote_bound）
  - 原文：EC based electrolytes have a side reaction (bubble) between 4.0 V to 4.5 V and mingle decomposition, while PC based electrolytes can be stable until about 4.2 V then slightly decompose

### 后续实验建议（尚未执行）
- 用硬碳（HC）||Na 半电池与 O3 型层状氧化物||HC 全电池，在 −20 °C（必要时扩展到 −10 与 0 °C）实测首周库仑效率、可逆容量与 100 周容量保持率，核对模型给出的 C5/C1 优先级是否在电池层面成立。
- 对 C5、C1（以及接近并列的 C3/C4）补测实测离子电导率与黏度的温度依赖曲线（−20 至 20 °C），用 Arrhenius 拟合给出表观活化能，验证低温排序并量化模型不确定度（RF 集成 log10 标准差 0.09–0.25，约对应 1.2–1.8 倍 1σ 区间）。
- 在 −20 °C 做 EIS 分析电荷转移电阻，并配合电解液去溶剂化能垒/溶剂化结构表征，区分是体相输运限制还是界面去溶剂化限制主导。
- 对高 EC 的 C1 与高 PC 的 C3/C4 分别做 LSV 阳极稳定性测试，确认在目标正极电压窗口内无 4.0–4.5 V 副反应。
- 在 −20 °C 与 20 °C 分别做倍率性能测试，确认室温电导率最优的 C1 在低温高倍率下是否仍优于 C5。
- 若后续考虑到成本与规模化，还需验证盐浓度较高的 C2 与低盐的 C5 在实用电解液用量与浸润性上的差异。