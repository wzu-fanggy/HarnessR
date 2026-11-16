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
- While the cells with EC based electrolytes almost do not operate below -20 °C (Fig. 3c). [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child5]（绑定级别：quote_bound）
  - 原文：While the cells with EC based electrolytes almost do not operate below -20 °C (Fig. 3c).
- all the values of conductivity decrease when the temperature drops (Fig. 2b) [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child2]（绑定级别：quote_bound）
  - 原文：all the values of conductivity decrease when the temperature drops (Fig. 2b)
- tend to experience a sharp decline in conductivity at low temperatures. [0086_Zhou 等 - 2025 - A Nonfluorinated Diluent in Ether-Based Electrolyte for Low-Temperature Sodium-Ion Batteries::sec1::child3]（绑定级别：quote_bound）
  - 原文：tend to experience a sharp decline in conductivity at low temperatures.
- the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity [0009_Cui 等 - 2025 - A Temperature-Adapted Ultraweakly Solvating Electrolyte for Cold-Resistant Sodium-Ion Batteries::sec0::child1]（绑定级别：quote_bound）
  - 原文：the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity
- Generally, low-temperature SIBs tend to suffer from slow ion transport properties. [0020_Feng 等 - 2024 - Monolithic Interphase Enables Fast Kinetics for High-Performance Sodium-Ion Batteries at Subzero Tem::sec3::child21]（绑定级别：quote_bound）
  - 原文：Generally, low-temperature SIBs tend to suffer from slow ion transport properties.

### 后续实验建议（尚未执行）
- 对 C5（及次优 C1）在 −20、−10、0、20 °C 用同一批配液、同一电导池（两阻塞电极交流阻抗）独立复测离子电导率：模型单点 1σ 不确定度约为 1.23–1.77 倍，C5 的 2.30 mS/cm 与 C1 的 1.42 mS/cm（−20 °C）差距在误差内需实测确认。
- C3 与 C4 的 −20 °C 预测值分别为 1.2419 与 1.2416 mS/cm，属统计并列（1σ 分别约 1.26 与 1.20 倍），建议并行测试；同时为 C4 增测低温外观、DSC 熔点与黏度，核查近纯 PC 配方（PC 占溶剂约 98.4%）在 −20 °C 是否出现结晶/相分离——文献报告 PC 基电解液在 −40 °C 电导率仅剩 0.693 mS/cm，模型对高 PC 端元的低温衰减可能估计不足。
- 在同一种硬碳（HC）负极、同一电芯构型（如 HC||Na 半电池或 NFM||HC 全电池）、同一倍率（如 0.2 C 与 1 C）与同一电压窗口（如 2.0–4.0 V 或 1.5–4.2 V，需在方案中统一写明）下比较 C5、C1 与对照配方的低温放电容量保持率、极化与阻抗，用电池级结果检验电导率初筛排序。
- 把 molar_ratios 换算为体积摩尔浓度（mol/L）后复测，确认极低盐摩尔分数配方（C5：0.085；C4：0.098）在实测中的真实盐浓度落在模型训练域内，避免组成表示差异引入偏差。
- 对 C2（−20 °C 预测仅 0.499 mS/cm，为五者最低，主因盐摩尔分数 0.200 最高、EC 仅 0.175）暂不进入首轮电池验证，除非低温复测证伪该预测。