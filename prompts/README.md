# Runtime Prompt

`prompts/runtime/` 保存已经审核并进入程序的系统 Prompt。组员候选版本仍放在 `collaboration/prompts/`。

当前版本：

- `interview_v2`：温情问询与半结构化筛查
- `companion_v2`：日常陪伴式聊天

发布流程：

1. 候选版本放入 `collaboration/prompts/candidates/`
2. 完成多轮案例、盲评和回归测试后放入 `collaboration/prompts/reviewed/`
3. 负责人确认后更新 `prompts/runtime/`
4. 更新版本号并运行全量对话测试
