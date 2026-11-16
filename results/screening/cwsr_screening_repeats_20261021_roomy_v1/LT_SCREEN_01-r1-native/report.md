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
- electrolytes play a pivotal role in governing low-temperature performance for three reasons: (1) they determine the concentration of free ions and their mobility, thereby regulating electrolyte conductivity; (2) their decomposition dictates the composition of the cathode/solid electrolyte interphase (CEI/SEI), thereby regulating the interphase resistance; (3) they determine the solvent-cation interaction, thereby regulating the de-solvation energy during the interfacial charge transfer reaction. [0054_Qin 等 - 2024 - Carboxylate ester-based electrolytes for Na-ion batteries::sec2::child2]（绑定级别：quote_bound）
  - 原文：electrolytes play a pivotal role in governing low-temperature performance for three reasons: (1) they determine the concentration of free ions and their mobility, thereby regulating electrolyte conductivity; (2) their decomposition dictates the composition of the cathode/solid electrolyte interphase (CEI/SEI), thereby regulating the interphase resistance; (3) they determine the solvent-cation interaction, thereby regulating the de-solvation energy during the interfacial charge transfer reaction.
- Classical carbonates, such as ethylene carbonate/propylene carbonate (EC/PC)-based electrolytes with high viscosity and strong solvation ability to $Na^{+}$ , tend to experience a sharp decline in conductivity at low temperatures. [0086_Zhou 等 - 2025 - A Nonfluorinated Diluent in Ether-Based Electrolyte for Low-Temperature Sodium-Ion Batteries::sec1::child3]（绑定级别：quote_bound）
  - 原文：Classical carbonates, such as ethylene carbonate/propylene carbonate (EC/PC)-based electrolytes with high viscosity and strong solvation ability to $Na^{+}$ , tend to experience a sharp decline in conductivity at low temperatures.
- Typically, cyclic PC and EC are of high dielectric constant and high viscosity, while linear EMC, DMC and DEC feature are of much lower viscosity and relatively lower dielectric constant (Table S1) [37]. [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child0]（绑定级别：quote_bound）
  - 原文：Typically, cyclic PC and EC are of high dielectric constant and high viscosity, while linear EMC, DMC and DEC feature are of much lower viscosity and relatively lower dielectric constant (Table S1) [37].
- finding that 1 M NaPF $_{6}$ offers the highest bulk conductivity. However, lower concentrations give comparable cycling performance in sodium-ion coin cells. [0100_0046_Properties_of_NaPFsub6sub_electroly::sec1::child0]（绑定级别：quote_bound）
  - 原文：finding that 1 M NaPF $_{6}$ offers the highest bulk conductivity. However, lower concentrations give comparable cycling performance in sodium-ion coin cells.
- EC has a melting point of 36 °C or approximately 309 K and as such is solid at 300 K, whereas the other solvents have melting points of -4.9 °C (PC) [241014689v1_Excess_Density_as_a_Descriptor_for_Electrolyte_Solvent_Design::sec2::child1]（绑定级别：quote_bound）
  - 原文：EC has a melting point of 36 °C or approximately 309 K and as such is solid at 300 K, whereas the other solvents have melting points of -4.9 °C (PC)
- However, at subzero temperatures, electrochemical reactions are significantly slower; it is believed that the $Na^{+}$ transport and desolvation process in the electrolyte can also dominate the electrode kinetics [10,27–29,48]. [0012_Deng 等 - 2022 - Self-optimizing weak solvation effects achieving faster low-temperature charge transfer kinetics for::sec5::child1]（绑定级别：quote_bound）
  - 原文：However, at subzero temperatures, electrochemical reactions are significantly slower; it is believed that the $Na^{+}$ transport and desolvation process in the electrolyte can also dominate the electrode kinetics [10,27–29,48].
- the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity. [0009_Cui 等 - 2025 - A Temperature-Adapted Ultraweakly Solvating Electrolyte for Cold-Resistant Sodium-Ion Batteries::sec0::child1]（绑定级别：quote_bound）
  - 原文：the voltage polarization on the HC surface at low temperatures (LT) increases dramatically, and results in a serious drop in battery capacity.
- For the low-temperature conductivity test, the cell was held in an oil bath capable of cooling down to -20 °C. [0054_Qin 等 - 2024 - Carboxylate ester-based electrolytes for Na-ion batteries::sec11::child1]（绑定级别：quote_bound）
  - 原文：For the low-temperature conductivity test, the cell was held in an oil bath capable of cooling down to -20 °C.
- But at a sub-zero temperature of $-20$ °C, the cells with PC electrolyte deliver significant polarization which further boosts at $-30$ °C. [0020_Feng 等 - 2024 - Monolithic Interphase Enables Fast Kinetics for High-Performance Sodium-Ion Batteries at Subzero Tem::sec3::child7]（绑定级别：quote_bound）
  - 原文：But at a sub-zero temperature of $-20$ °C, the cells with PC electrolyte deliver significant polarization which further boosts at $-30$ °C.
- When the temperature is lower than -20 °C, the ionic conductivity of CCE decreases rapidly, which can be attributed to the precipitation of $NaPF_{6}$ in the electrolyte. [0033_Li 等 - 2026 - Enhanced sodium storage in hard carbon via solvent co-intercalation electrolyte enabling Ah-level po::sec2::child6]（绑定级别：quote_bound）
  - 原文：When the temperature is lower than -20 °C, the ionic conductivity of CCE decreases rapidly, which can be attributed to the precipitation of $NaPF_{6}$ in the electrolyte.
- At low temperatures of -20 and -40 °C, the NTM-based electrolyte still delivers high discharge capacities of 71.5 mA h g $^{-1}$ and 62.2 mA h g $^{-1}$ , corresponding to 83.2% and 72.4% of the discharge capacity at room temperature, respectively (Figure 5g). [0004_Chang 等 - 2025 - Solvent-Fluorine-Free Anion-Enriched and Non-Flammable Electrolyte Enables Safe Sodium-Ion Battery::sec6::child3]（绑定级别：quote_bound）
  - 原文：At low temperatures of -20 and -40 °C, the NTM-based electrolyte still delivers high discharge capacities of 71.5 mA h g $^{-1}$ and 62.2 mA h g $^{-1}$ , corresponding to 83.2% and 72.4% of the discharge capacity at room temperature, respectively (Figure 5g).

### 后续实验建议（尚未执行）
- 在 −20/−10/0/20 °C 用不锈钢对称电池（SS|电解液|SS，玻璃纤维隔膜）配合油浴降温的 EIS 独立复测 C5、C1、C3、C4 的电导率，核对模型排序，并确认 C3 与 C4 在 −20 °C 的差异（约 0.0003 mS/cm）是否仅为模型噪声。
- 对优先候选 C5 与对照 C1、C3 在相同电极与电芯类型（如硬碳 || Na 半电池或硬碳 || 普鲁士白类全电池）、相同倍率与电压窗口下开展低温测试，报告 −20/−10/0 °C 放电容量相对室温的保持率、首周库仑效率与阻抗谱。
- 对高 PC 比例的 C3、C4 补测低温盐析出与结晶行为（目视/低温 DSC、降温后 EIS），验证是否在 −20 °C 附近出现电导率非 Arrhenius 突降。
- 对盐摩尔分数最高的 C2 补测黏度与离子对/溶剂化结构（Raman、变温电导率），确认其为五者最低电导率的结构原因，并判断是否值得保留在候选池中。
- 在电池级验证中同时评价室温（20 °C）性能，明确 C5 室温电导率并非最优但低温最优这一差异是否会在容量与倍率上被界面因素抵消。