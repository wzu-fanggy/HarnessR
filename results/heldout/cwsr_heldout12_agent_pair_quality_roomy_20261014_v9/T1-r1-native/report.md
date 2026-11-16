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
- Ionic conductivity is the topmost important property for an electrolyte and at the same time it should have negligible electronic conductivity in order to avoid short circuits. [0184_jijhydene201512090pdf::sec5::child0]（绑定级别：quote_bound）
  - 原文：Ionic conductivity is the topmost important property for an electrolyte and at the same time it should have negligible electronic conductivity in order to avoid short circuits.
- tend to experience a sharp decline in conductivity at low temperatures. [0086_Zhou 等 - 2025 - A Nonfluorinated Diluent in Ether-Based Electrolyte for Low-Temperature Sodium-Ion Batteries::sec1::child3]（绑定级别：quote_bound）
  - 原文：tend to experience a sharp decline in conductivity at low temperatures.
- at subzero temperatures, electrochemical reactions are significantly slower [0012_Deng 等 - 2022 - Self-optimizing weak solvation effects achieving faster low-temperature charge transfer kinetics for::sec5::child1]（绑定级别：quote_bound）
  - 原文：at subzero temperatures, electrochemical reactions are significantly slower
- subsequent electrochemical tests were based on preactivated HC anode in the voltage range from 1.0 to 3.8 V [0021_Gao 等 - 2025 - Constructing All-Climate Hybrid Sodium IonMetal Batteries through Intersolvent Synergistic Effect::sec1::child26]（绑定级别：condition_bound）
  - 原文：subsequent electrochemical tests were based on preactivated HC anode in the voltage range from 1.0 to 3.8 V

### 后续实验建议（尚未执行）
- 尚未执行：对优先候选 heldout-03、heldout-04、heldout-05、heldout-06 按给定摩尔比配制电解液，用同一电导率仪（同一电导池常数、同一升温/降温程序与热平衡时间）在 −20、−10、0、20 °C 各做独立电导率复测，并与本次 RF 预测值逐点比对（报告 na-conductivity-rf-v1 的 1σ 因子，约 1.10–1.80 倍），不得把预测值当作实测值。
- 尚未执行：在同一电极与同一电芯构型下做电池级验证——硬碳负极（HC）对 Na 金属半电池，以及 HC||NFPP（或 NFM||HC）全电池/软包，统一倍率（0.1C 及 0.3C）与统一电压窗口（如 HC 预钠化后 1.0–3.8 V），在 −20、−10、0、20 °C 测容量保持率、循环稳定性与 EIS 阻抗（分离体相与界面贡献），并与文献同构型同倍率基线比较。
- 尚未执行：对前四候选补做低温物性表征（黏度、DSC 玻璃化/凝固行为、电导率 Arrhenius 活化能），检验预测的低温电导率排序是否与实测物性一致，并据此判断是否需要调整盐浓度或 EC/EMC/PC 比例后再进入电池测试。