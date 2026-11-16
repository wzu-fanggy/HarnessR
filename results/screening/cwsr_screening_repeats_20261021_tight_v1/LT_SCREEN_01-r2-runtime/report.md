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
- they determine the concentration of free ions and their mobility, thereby regulating electrolyte conductivity [0054_Qin 等 - 2024 - Carboxylate ester-based electrolytes for Na-ion batteries::sec2::child2]（绑定级别：quote_bound）
  - 原文：they determine the concentration of free ions and their mobility, thereby regulating electrolyte conductivity
- tend to experience a sharp decline in conductivity at low temperatures. [0086_Zhou 等 - 2025 - A Nonfluorinated Diluent in Ether-Based Electrolyte for Low-Temperature Sodium-Ion Batteries::sec1::child3]（绑定级别：quote_bound）
  - 原文：tend to experience a sharp decline in conductivity at low temperatures.
- all the values of conductivity decrease when the temperature drops [0005_Che 等 - 2021 - Engineering optimization approach of nonaqueous electrolyte for sodium ion battery with long cycle l::sec8::child2]（绑定级别：quote_bound）
  - 原文：all the values of conductivity decrease when the temperature drops
- Because the viscosity effects are more pronounced at low temperature, the optimum EC composition locus is lower at low temperatures. [0404_Chayambuka 等 - 2021 - An experimental and modeling study of sodium-ion battery electrolytes::sec14::child1]（绑定级别：quote_bound）
  - 原文：Because the viscosity effects are more pronounced at low temperature, the optimum EC composition locus is lower at low temperatures.
- capacity retention only remains 50 % after 100 cycles, while the battery using 1 wt% NaDFP electrolyte shows better cycle stability and capacity retention remains 90 % after 100 cycles. [0002_Cai 等 - 2024 - A dual-functional electrolyte additive for stabilizing the solid electrolyte interphase and solvatio::sec7::child7]（绑定级别：quote_bound）
  - 原文：capacity retention only remains 50 % after 100 cycles, while the battery using 1 wt% NaDFP electrolyte shows better cycle stability and capacity retention remains 90 % after 100 cycles.
- the temperature that in practice sets the absolute lowest working and storage limit [0100_0046_Properties_of_NaPFsub6sub_electroly::sec1::child3]（绑定级别：quote_bound）
  - 原文：the temperature that in practice sets the absolute lowest working and storage limit

### 后续实验建议（尚未执行）
- 对 C5、C1、C3（并列第三的 C4 可一并）在本实验温度 −20、−10、0、20 °C 下用同一设备做独立电导率复测（冷却到温后保温稳定再测，建议交流阻抗法，给出每温度重复测量与误差），核对 na-conductivity-rf-v1 预测排序是否成立。
- 对优先候选 C5 与低温次优 C1 做 DSC/结晶温度与黏度–温度曲线测试，确认 −20 °C 下未发生冻结或结晶（PC 富集的 C4 尤其需要该项排查），这是模型无法给出的失效边界。
- 在相同电极、电芯类型、倍率与电压窗口下进行电池级对比：例如以硬碳为负极、相同正极组装同类电芯，在 −20 °C 与 20 °C 各做同一倍率下的容量、首周库仑效率与阻抗（EIS）对比，验证室温/低温容量保持差异。
- 对 C5 这类低 EC、富 EMC 配方补充界面验证（首周库仑效率、循环后阻抗与界面形貌/成分），确认低 EC 含量下 SEI 形成是否足以支撑低温循环；若无添加剂时界面劣化，应把界面调控作为并行测试项而非直接替换配方。
- 并行开展 −20 °C 长循环（≥100 圈）与倍率恢复测试，用于判断预测电导率优势是否会转化为实际低温容量保持。