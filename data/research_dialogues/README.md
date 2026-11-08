# 真实对话研究数据目录

该目录用于保存经过脱敏和知情同意的真实对话会话。不要把原始姓名、电话、地址、身份证号或未脱敏录音放入这里。

## 文件

- `dialogue_sessions_template.csv`：字段模板。
- `dialogue_sessions.csv`：实际导入后的会话表，默认被 `.gitignore` 排除。

## 字段

- `subject_id`：受试者级脱敏编号。
- `session_id`：会话编号，同一受试者可以有多条会话。
- `scale_name` / `scale_score`：PHQ-9、GDS-15、CESD-10 等量表名称与分数。
- `depression_label`：基于预先定义阈值的二分类标签。
- `transcript_json`：对话消息数组，格式同 `data/synthetic/dialogue_samples.csv`。
- `consent_code`：知情同意编号，不得填写姓名。
- `mpdd_risk`：如果该受试者有 MPDD 多模态风险概率，可填写；没有则留空。

## 导入与标注

```bash
python scripts/43_import_research_dialogues.py \
  --input /path/to/dialogue_sessions.csv
```

导入时会自动提取症状、持续时间、功能影响和安全风险等结构化对话特征。

## 模型比较

```bash
python scripts/44_compare_dialogue_models.py \
  --data data/research_dialogues/dialogue_sessions.csv
```

按受试者分组切分训练/验证/测试，比较：

1. 量表；
2. 量表 + 对话结构化线索；
3. 量表 + 对话 + MPDD 多模态风险。

真实数据不足时可以使用 `--demo` 运行流程烟测，但 demo 结果不能作为临床证据。
