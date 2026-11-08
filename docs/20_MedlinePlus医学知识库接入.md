# MedlinePlus 医学知识库接入

更新日期：2026-10-27

## 1 接入来源

本项目接入美国国家医学图书馆（NLM）公开的 MedlinePlus Health Topics Web Service。

官方页面：

- https://medlineplus.gov/about/developers/webservices/
- https://medlineplus.gov/about/using/citation/

使用条件：

- 免费且无需注册；
- 内容使用时明确标注来自 MedlinePlus.gov；
- 不使用 MedlinePlus Logo，不暗示官方背书；
- 请求频率不超过每分钟 85 次；
- NLM 建议缓存结果，当前实现采用离线长期缓存并按需手动刷新。

## 2 当前知识主题

当前缓存 18 个主题：

- Depression
- Anxiety
- Insomnia
- High blood pressure
- Diabetes
- Heart failure
- Stroke
- COPD
- Urinary incontinence
- Constipation
- Osteoarthritis
- Hypothyroidism
- Anemia
- Dehydration
- Falls
- Dementia
- Delirium
- Bereavement / grief

每个主题保存：

- 英文标题；
- 中文主题标签；
- 公开摘要；
- 来源机构；
- 官方页面 URL；
- 导入时间；
- 中文触发词。

## 3 数据流与隐私

导入阶段：

1. 运行 `scripts/48_import_medlineplus_kb.py`；
2. 脚本使用固定的英文医学主题词访问 MedlinePlus；
3. 不发送任何用户对话、年龄、性别或健康档案；
4. 仅保存公开健康主题的标题、短摘要和链接。

运行阶段：

1. Agent 根据中文触发词检索本地 JSON；
2. 不使用用户输入调用 MedlinePlus；
3. 命中的摘要进入本地规则和大模型提示；
4. 前端显示“MedlinePlus 参考”及官方链接；
5. 对话导出记录保存引用来源。

## 4 检索与使用边界

本地触发词只用于找到可能的患者教育主题。例如：

- “血压高”对应 High blood pressure；
- “突然糊涂、意识不清”对应 Delirium；
- “口干、尿少、腹泻”对应 Dehydration；
- “血糖高、口渴、尿多”对应 Diabetes。

检索命中不等于诊断。MedlinePlus 内容主要用于：

- 补充患者教育；
- 提醒需要追问的事实；
- 提供来源可追溯的健康信息；
- 帮助识别需要人工评估的情况。

系统不得：

- 根据一个关键词自动诊断；
- 把英文摘要整段照搬给老人；
- 用 MedlinePlus 内容替代医生、药师或急救；
- 把 MedlinePlus Logo 作为项目品牌；
- 声称 NLM 对本项目进行了认证或背书。

## 5 更新方法

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/48_import_medlineplus_kb.py
```

脚本串行请求并间隔 0.8 秒，低于官方每分钟 85 次限制。更新后会覆盖本地缓存，保留导入时间和官方 URL。

## 6 已知限制

- MedlinePlus 主要为英文患者教育资料，不是中国临床指南；
- 部分内容基于美国医疗体系和急救号码；
- 本项目只导入 18 个主题，不等于完整医学知识图谱；
- 摘要可能滞后于最新指南，需定期刷新并人工复核；
- 知识库检索不能替代医生问诊、查体和检查。

## 7 验证

自动测试覆盖：

- 缓存存在且包含来源和引用；
- 中英文主题映射；
- 血压、谵妄等主题检索；
- Agent 快照返回医学来源；
- 前端和导出记录显示来源；
- 伦理安全审计确认缓存包含来源和引用元数据。

若缓存缺失，Agent 仍可使用本地整理知识卡，不得现场抓取网页或编造来源。
