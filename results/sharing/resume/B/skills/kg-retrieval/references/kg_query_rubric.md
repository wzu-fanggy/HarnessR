# KG 查询质量检查表

## 实体规范化
- [ ] 使用规范名（如 NaPF6）而非中文全称
- [ ] 温度和条件只填关键词短语
- [ ] 不自行编造实体名

## 查询策略
- [ ] 结构化关系调用 `kg_query`
- [ ] 文本证据调用 `search_literature_chunks`
- [ ] 多角度扩展使用 `search_literature_multiview` 或 `explore_entity_neighborhood`
- [ ] 数值问题调用 `extract_numeric_properties`

## 引用纪律
- [ ] 只引用实际返回的文献
- [ ] 使用 `[chunk_id]` 格式
- [ ] 综述中转述需标注"转引自 X"
