# MedlinePlus 本地医学知识缓存

数据来源：MedlinePlus.gov，美国国家医学图书馆（NLM）。

官方服务：

- Web Service: https://medlineplus.gov/about/developers/webservices/
- Citation: https://medlineplus.gov/about/using/citation/

使用约束：

- 免费且无需注册；
- 标注内容来自 MedlinePlus.gov，不使用 MedlinePlus Logo；
- 不暗示 MedlinePlus 对本项目背书；
- Web Service 请求不超过每分钟 85 次；
- 生产环境仅读取本地缓存，运行时不会把用户对话发送给 MedlinePlus。

刷新缓存：

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/48_import_medlineplus_kb.py
```

缓存文件：

```text
data/medical_kb/medlineplus_topics.json
```

当前缓存包含 18 个主题：抑郁、焦虑、失眠、高血压、糖尿病、心力衰竭、卒中、COPD、尿失禁、便秘、骨关节炎、甲减、贫血、脱水、跌倒、痴呆、谵妄和哀伤反应。

该知识库是患者教育材料，不是临床指南、诊断模型或处方依据。
