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
- sodium ion batteries still suffer from severe performance degradation at low temperatures due to the conflict between ion desolvation and diffusion [0033_Li 等 - 2026 - Enhanced sodium storage in hard carbon via solvent co-intercalation electrolyte enabling Ah-level po::sec1::child0]（绑定级别：quote_bound）
  - 原文：sodium ion batteries still suffer from severe performance degradation at low temperatures due to the conflict between ion desolvation and diffusion
- the relatively high freezing point and high-polarity of carbonates cause a low ionic conductivity and slow charge transfer processes [0086_Zhou 等 - 2025 - A Nonfluorinated Diluent in Ether-Based Electrolyte for Low-Temperature Sodium-Ion Batteries::sec1::child2]（绑定级别：quote_bound）
  - 原文：the relatively high freezing point and high-polarity of carbonates cause a low ionic conductivity and slow charge transfer processes
- cyclic PC and EC are of high dielectric constant and high viscosity, while linear EMC, DMC and DEC feature are of much lower viscosity and relatively lower dielectric constant [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child0]（绑定级别：quote_bound）
  - 原文：cyclic PC and EC are of high dielectric constant and high viscosity, while linear EMC, DMC and DEC feature are of much lower viscosity and relatively lower dielectric constant

### 后续实验建议（尚未执行）
- 以电导池/EIS 在 −20、−10、0、20 °C 实测五个候选的离子电导率，校准 na-conductivity-rf-v1 预测并复算三温度最小值排序。
- 对 C5、C1 与 C3/C4 并行组装硬碳||Na 半电池（或 NFM/HC 全电池），在 −20/−10/0 °C 进行倍率与循环测试，报告相对室温容量的保持率。
- 用 DSC/目视考察低温相分离与结晶倾向：C1 的 EC 摩尔份额最高（0.569）、C4 的 PC 份额 0.888，均为低温析出高风险点。
- 补充低温界面表征（EIS 拆分体相电阻与界面电阻、SEI/CEI 成分），区分电导率不足与去溶剂化/界面阻抗两类限制。
- 目标成本尚未量化，需另行给出可接受的盐浓度与溶剂成本上限后再收敛配方。
- 确认候选的摩尔比例基准（本任务 molar_ratios 含盐）与实验配液用的体积/质量比之间的换算关系，避免配液偏差。