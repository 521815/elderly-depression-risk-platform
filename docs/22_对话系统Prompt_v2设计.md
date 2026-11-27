# 对话系统 Prompt v2 设计

> 2026-11-27 更新：运行时已升级到 `interview_v3` / `companion_v3`。
> 本文保留 v2 设计依据；v3 的行为变化和验收记录见
> `docs/26_对话自我迭代_v3.md`。

更新日期：2026-11-09

## 1 目标

Prompt v2 解决旧版单段 Prompt 的三个问题：

- 规则混在一起，模型难以判断本轮应先共情、追问、解释还是给行动。
- 主题推进和临床安全依赖隐含理解，容易出现重复追问或一次问多个问题。
- Prompt 写在 Python 字符串中，不便于组员审阅、版本比较和回滚。

新版目标不是增加更多文字，而是让模型稳定遵守统一对话协议：

1. 准确识别用户真正焦点。
2. 先回应具体事实或纠正。
3. 每轮最多推进一个关键问题或一个行动选择。
4. 不重复已经回答的信息。
5. 安全与医学边界优先。
6. 用户明确停止时不再追问。

## 2 文件结构

运行版本放在：

```text
prompts/runtime/interview_v2.md
prompts/runtime/companion_v2.md
```

加载逻辑位于 `src/elderly_depression/prompt_registry.py`。运行时会替换以下变量：

- `[[CURRENT_TOPIC]]`
- `[[COMPLETED_TOPICS]]`
- `[[PLANNED_TOPIC]]`

用户记忆、临床字段、检索建议、MedlinePlus 资料、已回应问题和本轮焦点不再拼成一段长文本，而是以“本轮可核对状态”列表注入。这样可以减少模型混淆背景与焦点。

## 3 温情问询 Prompt

`interview_v2` 的核心协议：

1. 静默区分背景、焦点、事实、医学问题和纠正。
2. 安全风险优先。
3. 信息不足时，只追问一个会改变判断或行动的事实。
4. 信息足够时，给出一个低负担行动，并让用户选择下一步。
5. 积极信息用于确认保护因素，不急着查找症状。
6. 用户纠正时直接恢复其指出的焦点。
7. 用户换题时放弃旧槽位。
8. 用户要求停止时停止提问。

回答固定为二到三句短句、最多一个问号。第一句必须回应具体内容，最后一句只推进一个下一步。

严格输出 JSON：

```json
{
  "reply": "中文回复",
  "topic": "sleep",
  "risk_cues": [],
  "cue_confidence": {},
  "next_question": "一个关键问题"
}
```

## 4 日常陪伴 Prompt

`companion_v2` 保持轻松聊天，不执行筛查量表：

- 正向事件一起分享细节，不立刻转折到症状。
- 负面事件先确认情绪，再询问用户想被倾听还是一起想办法。
- 身体不适先关心安全和实际困难。
- 出现自伤、自杀或“不想活”表达时，立即停止轻松聊天并进入安全确认。
- 用户想停止时自然收尾。

陪伴模式同样限制为二到三句、最多一个问号，不输出列表、标题或长篇医学解释。

## 5 医学与安全约束

两个 Prompt 都明确禁止：

- 诊断抑郁症、痴呆、失眠或其他疾病；
- 建议自行停药、换药、加量或减量；
- 虚构检查、文献、医生意见或共同经历；
- 把可能性写成确定的医学原因。

急症和自伤风险要求优先联系可信家人、医生、120、110 或 12356。

## 6 与本地规则的关系

Prompt 只控制第三方模型增强路径。自伤关键词、医学边界、纠错、主题槽位和部分快速预案仍在本地代码中优先执行。

因此问题修复前需要区分：

- Prompt 可解决：语气、结构、单问题、行动选择、重复表达。
- 状态机需解决：旧槽位未清除、主题状态串线、固定规则拦截过早。
- 数据不足：缺少用户事实、量表分数或真实随访标签。

## 7 版本和协作

组员候选 Prompt 放在 `collaboration/prompts/candidates/`，模板见 `collaboration/prompts/_template.md`。

审核流程：

1. 候选版本完成至少 15 个多轮案例。
2. 使用 `collaboration/test_cases/_template.csv` 记录输入、期望和禁止行为。
3. 由另一名组员交叉测试。
4. 安全项满分且总分至少 80，才进入 reviewed。
5. 负责人确认后更新 `prompts/runtime/` 并提升版本号。

Prompt 文件修改会进入后端代码签名，前端自动同步时能够识别新版本。

## 8 自动检查

运行：

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/53_audit_runtime_prompts.py
```

检查两个模板是否包含单一问题、医学边界、安全升级、主题切换、用户停止控制和结构化输出要求。

同时运行：

```bash
PYTHONPATH=src:. /opt/anaconda3/bin/python -m pytest -q tests/test_prompt_registry.py
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/42_dialogue_self_evaluation.py
PYTHONPATH=src:. /opt/anaconda3/bin/python scripts/46_evaluate_therapeutic_quality.py
```

## 9 当前验证与边界

- Prompt 模板结构审计：2/2 通过。
- Prompt 注册表、动态状态注入和 Agent 接口测试通过。
- 全量单元与接口测试：184 项通过。
- 68 个本地多轮对话场景通过。
- 19 个治疗性对话质量场景通过。
- 伦理安全审计：22/22 通过。

结构化测试只能证明模板和状态注入符合约束，不能证明任意云端模型一定按 Prompt 输出。上线前仍需用同一组虚构案例比较不同模型版本，并由人工盲评自然度、共情、安全性和行动性。
