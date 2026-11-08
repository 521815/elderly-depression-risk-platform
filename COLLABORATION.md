# 私有协作仓库规则

本仓库仅用于项目组内部代码和 Prompt 协作，不对外开源。数据集、论文、账号、模型和运行结果必须留在项目负责人的本机或学校受控存储中。

## 可以提交

- `backend_api/`、`src/`、`webapp/` 的源代码
- `scripts/` 中的可复现脚本
- `tests/` 中的测试
- `docs/` 中的 Markdown 方法说明
- Prompt 候选版本、虚构测试案例、评分和复盘
- `.env.example`、依赖和容器配置

## 禁止提交

- `.env` 和任何 API Key、归档密码
- `data/` 下的真实数据、派生特征、数据库和用户记录
- MPDD、CMDC、CHARLS 的原始或受许可约束文件
- `paper/`、论文导出文件和研究报告 Office 文件
- `output/` 下的模型、预测、日志、截图和实验结果
- 真实老人音频、视频、转写、姓名、联系方式和账号凭据

`.gitignore` 已默认阻止上述内容。不要使用 `git add -f` 绕过忽略规则。

## 分支和提交

- 主分支：`main`
- 功能分支：`feature/<模块>-<姓名>`
- Prompt 分支：`prompt/<主题>-<姓名>`
- 测试分支：`test/<主题>-<姓名>`
- 每人通过 Pull Request 提交，不直接修改 `main`
- 一个 Pull Request 只处理一个工作包

## Prompt 提交内容

每个 Prompt 工作包至少包含：

1. Prompt 目标和适用对话模式
2. 必须行为与禁止行为
3. 已知信息、槽位和输出格式
4. 至少 15 个多轮测试案例
5. 安全、共情、行动性、单问题和自然度评分
6. 与当前版本的差异和失败案例复盘

测试时使用虚构或充分脱敏的案例，不复制受限数据集中的原始转写。

## 代码提交前

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/pytest -q
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/47_ethics_security_audit.py
```

提交前必须确认：

- `git status --short` 中没有 `data/`、`output/`、`paper/` 或 `.env`
- 没有真实受试者信息
- 没有 API Key 或归档密码
- 测试和伦理审计通过
