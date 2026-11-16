# 输运性质脚本

Agent中优先使用原生工具：predict_formulation_conductivity、rank_formulations_under_uncertainty、evaluate_candidate_batch、inspect_candidate_artifact及compare_conductivity_predictions。生成大批候选时，generate_candidate_space显式设置result_mode="artifact"，保留真实batch_id。选择当前问题需要的操作，不要求固定工具序列。

独立兼容入口是 `electrolyte_design/scripts/evaluate_transport.py`，可通过 `run_script` 执行。共享运行图中predict/rank/evaluate须使用对应原生工具，以免独立子进程绕过资源租约；脚本请求不会被静默改写或重新运行。

- `predict`：明确组成和温度下的电导率预测。
- `rank`：任务已有**正的最低电导率要求**时的模型稳健性排序；没有此要求时不要填0、1或epsilon来制造阈值，可直接比较点预测。

调用参数：`--operation predict|rank|compare|generate|evaluate|inspect`，再选 `--request-json` 或 `--request-file`。
`--request-file` 只接受Orion工作区内路径；JSON对象沿用对应工具的 `request` 结构。
脚本的stdout是JSON结果，stderr/returncode由 `run_script` 包装返回。已知请求错误返回 `invalid_request`，退出码2；不能因此写成性能差。

预测示例请求（这是数学组成示例，不是文献实测或制备批量）：

```json
{"candidates":[{"candidate_id":"example","salt":"NaPF6","solvents":["EC","EMC","PC"],"additives":[],"component_mmol":{"NaPF6":1,"EC":4,"EMC":2,"PC":3}}],"temperature_c":20}
```

将请求JSON作为 `--request-json` 的一个完整参数传入；也可先写工作区文件再传路径。
排序请求的 `temperature_c` 可为1–6个不同温度，并须提供任务要求的正 `min_conductivity_ms_cm`；其余字段规则不变。

适用范围与解释：

- 当前模型仅覆盖NaPF6–EC/EMC/PC无添加剂及后端核查范围。域外或组成不足时保留反馈，不删添加剂、不猜密度。
- 盐摩尔分数分母是盐加全部溶剂；x_EC的分母是EC+EMC；x_PC的分母是全部溶剂。比例基准不能静默改变。
- 原文体积比不能仅改标签就当摩尔比。若另行提出摩尔配方，应明确是新设计，不继承原文配方实测性能。
- 摩尔分数不能单独确定mol/L；不需要mol/L的预测任务可以保留明确摩尔组成继续完成。
- 结果是模型估计，集成离散/联合满足比例不是已校准的实测可信度，不证明循环、安全或电极相容性。

运行环境由Orion项目Python及现有配置提供。Skill脚本以独立进程执行，当前不继承主图ToolNode的常驻资源租约；科学计算实现相同，但不能据此宣称内存/启动成本也相同。
现有独立工具接口暂保留兼容；加载本Skill不会自动从主工具表移除它们，当前尚不因此宣称已减少常驻工具上下文。


## 保存结果与同配方温度比较

每项新预测的 `result_binding` 给出 `record_id`、不可变组成身份、条件身份、实际模型版本和计算执行编号。完整记录保存于Orion工作区的 `reports/transport_predictions/`。`tool_run_id` 是工具执行编号；原生AI调用编号仍以对应ToolMessage的 `tool_call_id` 为准，可通过返回的record_id关联。

比较同配方的温度响应时，可用同一个脚本的 `--operation compare`，请求仅需：

```json
{"left_record_id":"左侧预测实际返回的record_id","right_record_id":"右侧预测实际返回的record_id"}
```

脚本读取保存记录，不接受在请求中重写预测值。只有组成和模型版本一致、温度不同、其它显式声明条件相同，才返回 `compared` 及右/左预测比值。`not_comparable` 是比较依据不成立，不代表某项配方性能差；若需温度对比，计算同一组成缺少的温度点。不能把旧C1的20°C值标成新C1r的20°C值，即使两个名称相近。

组成身份按实际模型输入的全组分摩尔分数建立，不随候选名称或整体用量同比例缩放改变，12位有效数字为本版本身份分辨率。配方改变会产生新身份，历史文件保留。条件身份至少包含温度；可通过预测请求的 `conditions` 对象保留其它显式条件，这些只是声明，并非模型已验证电极适配。

旧结果缺少记录、版本未知或文件校验不通过时，不补造身份或计算。当前只核查模型输入归属，不能据此声称原文v/v→mol/mol换算已核实，也不能保证自由文字不会绕开此比较接口。


## 大批候选的生成与评价

- `generate`：request沿用generate_candidate_space的盐/溶剂范围、n_samples、seed等参数。完整候选保存到工作区文件，只返回batch_id、空间统计和前两项预览。预览未预测，不是推荐。直接调用原生生成工具时也可显式设result_mode="artifact"；旧inline模式仍保留兼容。
- `evaluate`：读取batch_id并调用原有预测后端，默认评价该批全部候选，返回完整结果位置、实际覆盖数与点预测靠前项，不把几百条结果塞回上下文。
- `inspect`：给batch_id或evaluation_id（二选一），可按offset和limit分页读取完整保存记录。

批量评价请求示例：

```json
{"batch_id":"实际生成返回的64位标识","temperature_c":20,"objective":"maximise_predicted_conductivity","top_k":3}
```

`top_k`为摘要展示数（1–20）。`max_evaluations`可显式限制本轮评价预算；省略时评价全部生成项。若小于批次大小，工具只评价生成顺序的前一部分并明确返回unevaluated_count，不能据此声称筛完全部采样。当前单次至多5000项，与生成器上限一致。

`coverage`分别说明生成数、实际尝试数、成功预测数、未评价数及域外/输入不足等状态。不把域外项当零分。排名仅限实际成功预测的集合，连续组成空间尚未穷尽；采样覆盖率未知时保持未知。并列边界会报告未展示的并列数，ID排序只用于稳定展示，不提供统计显著性。

本操作只按给定温度的电导率点预测排序，不运行可配制性/全电池验证，也不引入最低电导率阈值。需要这些问题的证据或计算时由Agent另行选择适用工具。不要根据预览顺序任取前三个再声称已优化整批。

批量摘要对入选项同时给出all_component_mole_fractions（盐与全部溶剂为分母）、solvent_molar_shares（全部溶剂为分母）和实际model_features。原样使用各自定义，不把x_EC或1-x_EC标成三元溶剂内份额；离散度也只引用对应候选的记录。
inspect读取评价时默认predicted_rank顺序（失败项置后）；显式order="generation"可读取生成顺序，返回的order字段说明当前顺序。


## 改条件与跨批次身份

`candidate_id`（如gen-0200）只在一批中定位，换边界重新生成后会重用编号，不能据同名认定同配方，也不能说该编号在另一批必定不存在。跨批引用用 `candidate_ref`（batch_id:candidate_id），判断同一模型组成用composition_id。旧文件通过inspect读取时也返回限定到原批次的引用，不改写历史文件。

record_id标识一项具体计算，温度改变后重新计算应得到不同record_id。比较两个温度时，compare读取两个记录，核对composition_id与模型版本相同、温度不同；不是要求record_id相同。条件改变后保留旧计算及其旧范围，另检查新批次组成和温度，不把候选改名当成结果更新。
