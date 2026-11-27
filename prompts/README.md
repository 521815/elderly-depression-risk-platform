# Runtime Prompt

`prompts/runtime/` 保存已经审核并进入程序的系统 Prompt。组员候选版本仍放在 `collaboration/prompts/`。

当前版本：

- `interview_v3`：温情问询、半结构化筛查与用户主动选择支持方式
- `companion_v3`：日常陪伴、自然承接积极信息与低负担行动

`interview_v2` 和 `companion_v2` 保留用于行为对照和回滚。

发布流程：

1. 候选版本放入 `collaboration/prompts/candidates/`
2. 完成多轮案例、盲评和回归测试后放入 `collaboration/prompts/reviewed/`
3. 负责人确认后更新 `prompts/runtime/`
4. 更新版本号并运行全量对话测试
