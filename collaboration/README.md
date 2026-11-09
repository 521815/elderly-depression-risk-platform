# Prompt 协作区

该目录用于组员提交和评审 Prompt，不包含应用运行数据。

组员入门材料：

- `handbook/老年抑郁风险筛查对话智能体Prompt协作手册.docx`
- 仓库根目录 `COLLABORATION.md`
- `prompts/_template.md`
- `test_cases/_template.csv`

```text
collaboration/
├── prompts/
│   ├── candidates/
│   ├── reviewed/
│   └── approved/
├── test_cases/
├── reviews/
└── meeting_notes/
```

- `candidates/`：组员原始候选版本
- `reviewed/`：经过交叉测试、仍需修改的版本
- `approved/`：负责人确认可进入工程的版本
- `test_cases/`：虚构或脱敏后的多轮案例
- `reviews/`：盲评、评分和失败分析
- `meeting_notes/`：任务分工与决策记录

文件命名格式：

```text
prompt_<主题>_<姓名>_v0.1.md
cases_<主题>_<姓名>_v0.1.csv
review_<主题>_<姓名>_v0.1.md
```
