## 电解液配方筛选

### 候选组成
- heldout-01：全组分摩尔比例为 NaPF6=0.051698，EC=0.00808389，EMC=0.00892642，PC=0.931292
- heldout-02：全组分摩尔比例为 NaPF6=0.0539646，EC=0.00191335，EMC=0.0429229，PC=0.901199
- heldout-03：全组分摩尔比例为 NaPF6=0.057711，EC=0.167813，EMC=0.262776，PC=0.511699
- heldout-04：全组分摩尔比例为 NaPF6=0.0852192，EC=0.173613，EMC=0.736773，PC=0.00439533
- heldout-05：全组分摩尔比例为 NaPF6=0.095493，EC=0.137906，EMC=0.170677，PC=0.595924
- heldout-06：全组分摩尔比例为 NaPF6=0.0993705，EC=0.424667，EMC=0.196835，PC=0.279128
- heldout-07：全组分摩尔比例为 NaPF6=0.124383，EC=0.277281，EMC=0.146365，PC=0.451972
- heldout-08：全组分摩尔比例为 NaPF6=0.136592，EC=0.0991394，EMC=0.221273，PC=0.542996
- heldout-09：全组分摩尔比例为 NaPF6=0.159315，EC=0.00392615，EMC=0.0892349，PC=0.747523
- heldout-10：全组分摩尔比例为 NaPF6=0.18975，EC=0.120062，EMC=0.236559，PC=0.453629
- heldout-11：全组分摩尔比例为 NaPF6=0.195695，EC=0.0245508，EMC=0.737024，PC=0.0427303
- heldout-12：全组分摩尔比例为 NaPF6=0.19824，EC=0.140042，EMC=0.0997755，PC=0.561942

### 电导率预测
模型：na-conductivity-rf-v1。数值单位为 mS/cm。

| 候选 | -20 °C | -10 °C | 0 °C | 20 °C |
| --- | --- | --- | --- | --- |
| heldout-01 | 1.6693 | 2.5318 | 3.5213 | 5.7022 |
| heldout-02 | 1.7251 | 2.5507 | 3.5255 | 6.4185 |
| heldout-03 | 2.4200 | 3.5385 | 4.6338 | 7.1028 |
| heldout-04 | 2.0513 | 2.9930 | 3.9970 | 7.0165 |
| heldout-05 | 1.9745 | 2.9763 | 4.0811 | 6.5059 |
| heldout-06 | 1.9566 | 3.0328 | 4.4605 | 7.2281 |
| heldout-07 | 1.2463 | 2.0487 | 3.1548 | 6.5592 |
| heldout-08 | 1.3414 | 2.3762 | 3.3288 | 6.1227 |
| heldout-09 | 0.6722 | 1.2964 | 2.1239 | 4.6370 |
| heldout-10 | 0.4193 | 0.9259 | 1.7025 | 4.2832 |
| heldout-11 | 0.7529 | 1.3566 | 2.3077 | 5.1101 |
| heldout-12 | 0.3299 | 0.8240 | 1.4855 | 3.3674 |

### 初筛结果
在 -20、-10、0 °C 的 na-conductivity-rf-v1 模型预测中，各候选最低电导率为 heldout-01=1.66929、heldout-02=1.7251、heldout-03=2.41997、heldout-04=2.05128、heldout-05=1.97452、heldout-06=1.95663、heldout-07=1.24634、heldout-08=1.34138、heldout-09=0.672235、heldout-10=0.419253、heldout-11=0.752933、heldout-12=0.329873 mS/cm。按最低预测电导率最大这一初筛指标，优先开展后续实验的前4种候选依次为heldout-03（2.4200 mS/cm）、heldout-04（2.0513 mS/cm）、heldout-05（1.9745 mS/cm）、heldout-06（1.9566 mS/cm）。这一排序来自模型计算，不等于电池容量或循环性能的实验排名。在参照温度 20 °C，预测值为 heldout-01=5.70221、heldout-02=6.41849、heldout-03=7.10284、heldout-04=7.01649、heldout-05=6.50593、heldout-06=7.22806、heldout-07=6.55923、heldout-08=6.12269、heldout-09=4.63697、heldout-10=4.28318、heldout-11=5.11008、heldout-12=3.36744 mS/cm，数值最高的候选为 heldout-06。

### 文献依据
## 声明绑定状态（语义蕴含仍未自动验证）
- electrolytes play a pivotal role in governing low-temperature performance for three reasons: (1) they determine the concentration of free ions and their mobility, thereby regulating electrolyte conductivity [0054_Qin 等 - 2024 - Carboxylate ester-based electrolytes for Na-ion batteries::sec2::child2]（绑定级别：quote_bound）
  - 原文：electrolytes play a pivotal role in governing low-temperature performance for three reasons: (1) they determine the concentration of free ions and their mobility, thereby regulating electrolyte conductivity
- Ionic conductivity is the topmost important property for an electrolyte and at the same time it should have negligible electronic conductivity in order to avoid short circuits [0184_jijhydene201512090pdf::sec5::child0]（绑定级别：quote_bound）
  - 原文：Ionic conductivity is the topmost important property for an electrolyte and at the same time it should have negligible electronic conductivity in order to avoid short circuits
- The good power capability of these two cells can be attributed to high ionic conductivity of the electrolytes, which results in lower charging and discharge over-potentials [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child3]（绑定级别：quote_bound）
  - 原文：The good power capability of these two cells can be attributed to high ionic conductivity of the electrolytes, which results in lower charging and discharge over-potentials
- However, at subzero temperatures, electrochemical reactions are significantly slower; it is believed that the [0012_Deng 等 - 2022 - Self-optimizing weak solvation effects achieving faster low-temperature charge transfer kinetics for::sec5::child1]（绑定级别：quote_bound）
  - 原文：However, at subzero temperatures, electrochemical reactions are significantly slower; it is believed that the
- At low temperatures, the ionic conductivity in bulk electrolyte diminishes, and the interfacial charge transfer becomes somewhat restricted, which in turn impacts the efficiency during discharging and charging [0029_Li 等 - 2024 - The ethers chain length effect in electrolyte for hard carbon towards efficient sodium storage at l::sec5::child18]（绑定级别：quote_bound）
  - 原文：At low temperatures, the ionic conductivity in bulk electrolyte diminishes, and the interfacial charge transfer becomes somewhat restricted, which in turn impacts the efficiency during discharging and charging
- sodium ion kinetics are affected not only by bulk diffusion within the electrolyte but also intimately linked to transport across the EEI and the desolvation process [0199_Liu 等 - 2026 - Mechanism of gas generation and a high stable SEI construction with PCDFEC-based electrolyte for so::sec9::child1]（绑定级别：quote_bound）
  - 原文：sodium ion kinetics are affected not only by bulk diffusion within the electrolyte but also intimately linked to transport across the EEI and the desolvation process
- The low-temperature environment hinders the ion transport and causes a significant increase of internal resistance in the cell, ultimately leading to a large voltage polarization and great downward trend in capacity [0086_Zhou 等 - 2025 - A Nonfluorinated Diluent in Ether-Based Electrolyte for Low-Temperature Sodium-Ion Batteries::sec1::child4]（绑定级别：quote_bound）
  - 原文：The low-temperature environment hinders the ion transport and causes a significant increase of internal resistance in the cell, ultimately leading to a large voltage polarization and great downward trend in capacity
- current SIBs are still plagued by severe parasitic reactions at high temperature (HT) and sluggish ionic transfer kinetics at low temperature (LT) [0021_Gao 等 - 2025 - Constructing All-Climate Hybrid Sodium IonMetal Batteries through Intersolvent Synergistic Effect::sec1::child1]（绑定级别：quote_bound）
  - 原文：current SIBs are still plagued by severe parasitic reactions at high temperature (HT) and sluggish ionic transfer kinetics at low temperature (LT)

### 后续实验建议（尚未执行）
- 待执行（实验一，独立电导率复测）：对 4 个优先候选 heldout-03、-04、-05、-06 以及 20 °C 参照下进入前四的 heldout-07，按输入摩尔比在手套箱内独立配制 NaPF6–EC/EMC/PC 电解液，用同一电导池、同一台阻抗谱仪在 −20、−10、0、20 °C 恒温平衡后各重复测量 3 次，报告均值与标准差；与本轮 48 个 na-conductivity-rf-v1 预测点逐点比对，重点检验差异远小于模型不确定度的 heldout-05（−20 °C 1.975 mS/cm）与 heldout-06（1.957 mS/cm）之间的排序是否真实。
- 待执行（实验二，电池级验证）：在电导率复测通过后，用同一电极、同一电芯构型、同一倍率与同一电压范围测试上述候选：硬碳负极为同一批次，正极取同一批次 Na 金属（HC‖Na 半电池，0.01–1.5 V）或同一批次层状氧化物/磷酸盐正极（全电池），装配同规格 2032 型扣式电池或同规格单层软包；在 0.2 C 与 1 C 下测 −20、−10、0、20 °C 的放电容量、相对 20 °C 的容量保持率、循环稳定性与阻抗谱，只有在条件完全一致时才做候选间比较。
- 待执行（实验三，机理与界面归因）：对电导率复测最优但电池容量偏低的情形，按文献做法用变温 EIS/Arrhenius 分析区分跨电极/电解液界面输运与脱溶剂化两类能垒（参见 [0199_Liu 等 - 2026 - Mechanism of gas generation and a high stable SEI construction with PCDFEC-based electrolyte for so::sec9::child1]），并检查 PC 主导候选（heldout-01/-02/-09）是否出现盐析出或低温结晶。
- 待评估（近似工况与成本）：本轮初筛仅在 −20/−10/0/20 °C 的电导率维度评分，未覆盖真实应用中的其他工况（如更快的 2–5 C 充电、更高电压上限、长循环）与成本目标；本轮亦未评估 EC/EMC/PC 的相对用量成本与供应可得性，这些目标需在完成上述电导率复测与电池级测试后再做定量评估。