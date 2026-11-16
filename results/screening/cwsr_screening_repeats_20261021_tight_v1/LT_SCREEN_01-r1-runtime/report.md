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
- Ionic conductivity is critical to evaluate the performance of electrolytes, especially at low temperatures [0038_Liu 等 - 2024 - Ester-based anti-freezing electrolyte achieving ultra-low temperature cycling for sodium-ion batteri::sec10::child4]（绑定级别：quote_bound）
  - 原文：Ionic conductivity is critical to evaluate the performance of electrolytes, especially at low temperatures
- all the values of conductivity decrease when the temperature drops [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child2]（绑定级别：quote_bound）
  - 原文：all the values of conductivity decrease when the temperature drops
- mainly caused by the salt precipitation or the freezing of the electrolytes [220902409v1_Ultra-Low_Temperature_Li_CFx_Batteries_Enabled_by_Fast-transport_and_Anion-pairi::sec3::child3]（绑定级别：quote_bound）
  - 原文：mainly caused by the salt precipitation or the freezing of the electrolytes
- the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity [0009_Cui 等 - 2025 - A Temperature-Adapted Ultraweakly Solvating Electrolyte for Cold-Resistant Sodium-Ion Batteries::sec0::child1]（绑定级别：quote_bound）
  - 原文：the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity
- This allows the electrolyte to have faster sodium ion transport kinetics at lower temperatures, thereby ensuring the normal operation of the sodium ion battery at low temperatures [0028_Li 等 - 2024 - Hierarchical doping electrolyte solvation engineering to achieve high-performance sodium-ion batteri::sec11::child5]（绑定级别：quote_bound）
  - 原文：This allows the electrolyte to have faster sodium ion transport kinetics at lower temperatures, thereby ensuring the normal operation of the sodium ion battery at low temperatures
- The good power capability of these two cells can be attributed to high ionic conductivity of the electrolytes, which results in lower charging and discharge over-potentials [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child3]（绑定级别：quote_bound）
  - 原文：The good power capability of these two cells can be attributed to high ionic conductivity of the electrolytes, which results in lower charging and discharge over-potentials
- electrolyte of EC: PC = 7:3 composition shows peak in the transport properties among the electrolytes studied within the MD method [0263_Chandel 等 - 2024 - Multiscale Molecular Dynamics and Electrochemical Modelling of Sodium-Ion Batteries Impact of Elect::sec12::child2]（绑定级别：quote_bound）
  - 原文：electrolyte of EC: PC = 7:3 composition shows peak in the transport properties among the electrolytes studied within the MD method

### 后续实验建议（尚未执行）
- 对初筛首位候选 C5（NaPF6 0.0852 / EC 0.1736 / EMC 0.7368 / PC 0.0044 摩尔比）在 −20、−10、0 °C 用同一批电解液做独立电导率复测（两电极阻塞电池 + 交流阻抗，每温度至少 3 次平行），检验 na-conductivity-rf-v1 的预测值 2.30 / 3.18 / 3.88 mS/cm 与 1σ 区间（约 ±1.23 倍）是否成立。
- 同批复测 20 °C 电导率，用于确认室温排序（C1 > C3 > C5 ≈ C4 > C2）与低温初筛排序（C5 > C1 > C3 ≈ C4 > C2）确实不一致，避免用室温单点数据代替低温初筛。
- 并行测试 C1 与 C3/C4：C1 的 −20 °C 预测值（1.42 mS/cm）居第二但 1σ 约 ±1.55 倍，C3 与 C4 的 −20 °C 预测值几乎相同（1.242 与 1.242 mS/cm），属于模型无法区分的并列区间，需以实测区分而不宜仅凭排序淘汰。
- 在相同电极与电芯构型（如 HC || Na 半电池或 HC || 层状氧化物全电池）、相同电压窗口与倍率下，比较 C5、C1、C3 在 −20 °C 的容量发挥、循环保持率与 EIS（体相电阻、电荷转移电阻、SEI 电阻），验证电导率优势是否转化为可用容量。
- 对 PC 高占比候选（C3、C4）与 C5 补充 DSC/目视低温静置试验，检查是否出现盐析出或电解液冻结——文献显示低温电导率骤降常由盐析出或冻结而非仅黏度引起；若 C5 出现冻结/析出，则改以 C1 为主并并行测试 C3。