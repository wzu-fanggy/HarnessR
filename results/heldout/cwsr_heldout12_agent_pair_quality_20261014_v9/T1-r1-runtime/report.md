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
- Generally, low-temperature SIBs tend to suffer from slow ion transport properties. [0020_Feng 等 - 2024 - Monolithic Interphase Enables Fast Kinetics for High-Performance Sodium-Ion Batteries at Subzero Tem::sec3::child21]（绑定级别：quote_bound）
  - 原文：Generally, low-temperature SIBs tend to suffer from slow ion transport properties.
- So the ionic conductivity of five types of electrolyte was detected, and as expected, all the values of conductivity decrease when the temperature drops (Fig. 2b). [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child2]（绑定级别：quote_bound）
  - 原文：So the ionic conductivity of five types of electrolyte was detected, and as expected, all the values of conductivity decrease when the temperature drops (Fig. 2b).
- However, the accumulation of inorganic components in the inner solid electrolyte interphase (SEI), caused by substantial sodium salt decomposition, impedes the kinetics of $Na^{+}$ interphasial transport. [0013_Deng 等 - 2025 - Heterostructured Solid Electrolyte Interphase Enables Facilitated Kinetics for Low-Temperature Sodiu::sec0::child0]（绑定级别：quote_bound）
  - 原文：However, the accumulation of inorganic components in the inner solid electrolyte interphase (SEI), caused by substantial sodium salt decomposition, impedes the kinetics of $Na^{+}$ interphasial transport.
- Thus, the monolithic electrode/electrolyte interphase with more organic substances can provide favorable $Na^{+}$ transport kinetics, which explains why 0.5M-D helps SIBs maintain excellent performance at low temperature. [0020_Feng 等 - 2024 - Monolithic Interphase Enables Fast Kinetics for High-Performance Sodium-Ion Batteries at Subzero Tem::sec3::child21]（绑定级别：quote_bound）
  - 原文：Thus, the monolithic electrode/electrolyte interphase with more organic substances can provide favorable $Na^{+}$ transport kinetics, which explains why 0.5M-D helps SIBs maintain excellent performance at low temperature.
- For example, when using $NaPF_{6}$ electrolyte in EC:DMC (3:7 wt%) between 0.4 M and 1 M, the 0.6 M solution gave the highest conductivity (6.8 mS cm $^{-1}$ ). [0100_0046_Properties_of_NaPFsub6sub_electroly::sec1::child4]（绑定级别：quote_bound）
  - 原文：For example, when using $NaPF_{6}$ electrolyte in EC:DMC (3:7 wt%) between 0.4 M and 1 M, the 0.6 M solution gave the highest conductivity (6.8 mS cm $^{-1}$ ).
- Combined with the above characteristics, the PEMP electrolyte designed in this work exhibits a high ionic conductivity (5.75 mS cm $^{-1}$ ) at -20 °C. [0028_Li 等 - 2024 - Hierarchical doping electrolyte solvation engineering to achieve high-performance sodium-ion batteri::sec12::child1]（绑定级别：quote_bound）
  - 原文：Combined with the above characteristics, the PEMP electrolyte designed in this work exhibits a high ionic conductivity (5.75 mS cm $^{-1}$ ) at -20 °C.
- Additionally, the $Na_{0.85}Li_{0.12}Ni_{0.22}Mn_{0.66}O_{2}||HC$ full cells exhibit a high discharge capacity of 85 mAh g $^{-1}$ at -20 °C (85% of its room-temperature capacity). [0013_Deng 等 - 2025 - Heterostructured Solid Electrolyte Interphase Enables Facilitated Kinetics for Low-Temperature Sodiu::sec0::child1]（绑定级别：quote_bound）
  - 原文：Additionally, the $Na_{0.85}Li_{0.12}Ni_{0.22}Mn_{0.66}O_{2}||HC$ full cells exhibit a high discharge capacity of 85 mAh g $^{-1}$ at -20 °C (85% of its room-temperature capacity).
- The results of DCR test at $-20^{\circ}C$ shows (Figure S9d) that the DCR value of the battery with 0.5 M NaFSI double-salt system is significantly reduced, which is the key reason why it can release more capacity at low temperatures. [0015_Fan 等 - 2024 - A Sodium Bisfluorosulfonylimide NaFSI-based Multifunctional Electrolyte Stabilizes the Performan::sec8::child9]（绑定级别：quote_bound）
  - 原文：The results of DCR test at $-20^{\circ}C$ shows (Figure S9d) that the DCR value of the battery with 0.5 M NaFSI double-salt system is significantly reduced, which is the key reason why it can release more capacity at low temperatures.
- These all can be speculated that the polarization is related to the large impedance of the electrode interface. [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child3]（绑定级别：quote_bound）
  - 原文：These all can be speculated that the polarization is related to the large impedance of the electrode interface.

### 后续实验建议（尚未执行）
- 尚未执行：对全部 12 种候选（优先 heldout-03、heldout-04、heldout-05、heldout-06）在 -20、-10、0、20 °C 做独立离子电导率复测——同一批 NaPF6 与 EC/EMC/PC 原料、同一电导池与同一升温/降温程序、同一水分与氧含量控制条件，并与本报告记录的预测值逐点比对，确认模型排序在实测中是否成立（本轮预测未经任何留出实测真值检验）。
- 尚未执行：在同等电极与电芯构型（同一正极、同一硬碳负极、同一隔膜与电解液用量，例如 HC||Na 半电池或 NVP||HC 全电池）、相同倍率（如 0.2C 充放电）与相同电压窗口（如 2.0–4.0 V）下，对同一批四种优先候选在 -20、-10、0、20 °C 测试放电容量、容量保持率、库仑效率与阻抗（EIS/DCR），检验低温最低预测电导率排序是否与电池低温容量排序一致；本报告未做任何电池测试。
- 尚未执行：补充电化学稳定窗口（LSV）与铝箔/壳体腐蚀、钠沉积形态、首效及 N/P 比的表征，因为高盐或 PC 富集配方在低温下的界面副反应无法由电导率预测判断。
- 尚未执行：量化仍未评估的应用目标，包括低温下的成本、黏度与凝固点，以及 -40 °C 及更低温度的可用性；本报告的排序只覆盖 -20、-10、0 °C 的预测电导率。