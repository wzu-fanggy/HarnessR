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
- Despite the approximate 50% lower bulk conductivity of the 0.25 M electrolyte compared to the 1 M electrolyte, the discharge capacities are not affected, even at the moderately high rates of 2C. [0100_0046_Properties_of_NaPFsub6sub_electroly::sec1::child11]（绑定级别：quote_bound）
  - 原文：Despite the approximate 50% lower bulk conductivity of the 0.25 M electrolyte compared to the 1 M electrolyte, the discharge capacities are not affected, even at the moderately high rates of 2C.
- With the decrease in temperature, the ionic conductivity of the PC-based electrolyte drops sharply [0038_Liu 等 - 2024 - Ester-based anti-freezing electrolyte achieving ultra-low temperature cycling for sodium-ion batteri::sec10::child4]（绑定级别：quote_bound）
  - 原文：With the decrease in temperature, the ionic conductivity of the PC-based electrolyte drops sharply
- The viscosity is a crucial property of the electrolyte as highly viscous solutions will hinder the movement of ions and lower the conductivity. [0100_0046_Properties_of_NaPFsub6sub_electroly::sec1::child6]（绑定级别：quote_bound）
  - 原文：The viscosity is a crucial property of the electrolyte as highly viscous solutions will hinder the movement of ions and lower the conductivity.

### 后续实验建议（尚未执行）
- 对 C5（初筛第一）以及 C1、C3、C4（低温接近或并列）在 −20/−10/0 °C 做独立电导率实测复测：同一批 NaPF6 与溶剂、统一水含量（如 <20 ppm）与同一电导池/EIS 方法，至少 3 次重复，用以检验模型预测及 1σ 量级的差距是否可分辨。
- 对 C5 与 C1 在相同电极（如硬碳 || Na 半电池以及硬碳 || 正极全电池）、相同电芯形式、相同倍率与相同电压窗口下比较实际可逆容量、−20 °C 循环稳定性与阻抗（EIS），不能把预测电导率排序当作电池性能结论。
- 补充低温凝固点与黏度表征（DSC/流变/目视凝固），因为 C4（PC 摩尔占比 0.888）与 C3（PC 0.543）等 PC 富配方的模型电导率较高，但模型未表征凝固与相分离风险。
- 对 C3 与 C4 的 −20 °C 预测值几乎相同（1.242 vs 1.242 mS/cm）且不确定度区间重叠，建议并行测试而不是排序取舍；同时记录实测值与预测值的偏差用于模型校准。