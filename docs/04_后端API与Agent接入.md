# 后端 API 与 Agent 接入

## 启动方式

桌面入口 `~/Desktop/老年抑郁风险筛查.command` 会同时启动：

- 后端 API：`http://localhost:8001`
- 前端工作台：`http://localhost:8501`

也可以手动启动：

```bash
python backend_api/app.py
streamlit run webapp/app.py
```

## 主要接口

### 认证与账号

- `GET /api/auth/challenge`：获取登录/注册用的随机加减法人机验证题
- `POST /api/auth/login`：登录，body 为 `{"username": "...", "password": "...", "challenge_id": "...", "answer": 12}`
- `POST /api/auth/register`：注册普通用户并登录，字段同登录，另可传 `display_name`
- `GET /api/auth/me`：读取当前登录账号
- `POST /api/auth/logout`：注销当前会话
- `GET /api/auth/security-options`：读取注册可选密保问题
- `POST /api/auth/recovery/questions`：根据用户名读取密保问题
- `POST /api/auth/recovery/reset`：回答全部密保问题后重置密码
- `PATCH /api/account/profile`：登录后修改自己的用户名、显示名称、密码和密保问题
- `GET /api/admin/accounts`：管理员查看账号列表
- `POST /api/admin/accounts`：管理员创建账号，body 含 `username`、`password`、`role`、`display_name`
- `PATCH /api/admin/accounts/<account_id>`：管理员修改角色、状态、显示名称或密码

除状态接口与登录注册接口外，其余 API 都需要请求头 `Authorization: Bearer <token>`。普通用户只能访问自己的档案，管理员接口会在后端校验角色。

### 数据与训练

- `GET /api/status`：数据与模型状态
- `GET /ready`：应用就绪检查，验证认证库、用户记录库和输出目录是否可读可写
- `GET /api/models/list`：已保存的模型文件列表
- `GET /api/models/registry`：模型注册表（模型、指标、路径）
- `GET /api/models/cards`：CHARLS、MPDD、CMDC、融合层和模拟组的模型卡，包含适用人群、角色、指标、禁止用途、报告 SHA-256 和数据指纹
- `GET /api/research/incremental_validation`：读取 CHARLS 理论基线、静态共病与共病轨迹对照、MPDD 同人级模态消融和跨源证据审计
- `POST /api/research/incremental_validation`：重新运行 B0-B4T 基线、15 种模态组合消融、配对比较和跨源审计
- `POST /api/jobs`：提交后台任务；相同任务和参数的运行中实例会复用，返回 `deduplicated=true`
- `GET /api/jobs/<job_id>`：查询后台任务状态
- `POST /api/jobs/<job_id>/cancel`：请求取消任务；取消或超时后，后台线程不能覆盖最终状态
- 后台任务支持 `cmdc_train`、`cmdc_advanced`、`cmdc_holdout`、`modma_audit`、`modma_audio` 和 `modma_transfer`，长分析不再阻塞同步 API
- `GET /api/charls/summary`：CHARLS 标准表概览
- `POST /api/charls/build`：构建 CHARLS 标准表
- `POST /api/charls/train`：训练表格基线，body 为 `{"resample_method": "smote"}`
- `POST /api/charls/train_temporal`：时间外验证，前波次训练、后波次验证，body 为 `{"resample_method": "smote_tomek"}`
- `GET /api/charls/temporal_report`：读取时间外验证报告
- `POST /api/charls/train_longitudinal`：个体纵向轨迹，上一波次特征预测当前波次，按人分组切分
- `GET /api/charls/longitudinal_report`：读取纵向轨迹报告
- `POST /api/imbalance/compare`：运行 6 种类别不平衡方法对比
- `POST /api/fusion/train`：训练学习型融合权重，body 为 `{"rows": [...], "label_col": "depression"}`
- `GET /api/fusion/report`：读取融合模型报告
- `GET /api/fusion/status`：读取当前融合模式（learned / literature_prior）
- `GET /api/mpdd/summary`：MPDD 数据状态
- `POST /api/mpdd/build`：扫描 `data/mpdd` 官方 `.npy` 特征并生成 `data/mpdd_features.csv`
- `POST /api/mpdd/build_utterances`：生成 utterance 级特征表 `data/mpdd_utterance_features.csv`
- `POST /api/mpdd/build_cache`：生成分模态 Parquet 缓存和 manifest
- `GET /api/mpdd/cache_status`：查看原始/CSV/Parquet 大小与缓存状态
- `POST /api/mpdd/train_subject`：训练被试级分组重复交叉验证基线并保存推理管线
- `GET /api/mpdd/subject_report`：读取被试级基线汇总报告
- `POST /api/mpdd/train_advanced`：训练 label3 三分类和 PHQ-9 回归评估
- `GET /api/mpdd/advanced_report`：读取三分类/回归报告
- `POST /api/mpdd/export_test_predictions`：导出测试集 23 名被试的预测结果
- `POST /api/mpdd/upload`：multipart 上传 MPDD 合并特征 CSV
- `POST /api/mpdd/train`：训练多模态后期融合模型
- `GET /api/cmdc/summary`：CMDC 受试者、模态覆盖、合规和外部验证状态
- `POST /api/cmdc/build`：从外部加密归档生成去标识化特征表；密码只在单次请求内使用
- `POST /api/cmdc/train`：按受试者运行 CMDC 重复分层交叉验证并保存独立模型
- `GET /api/cmdc/report`：读取 CMDC 分模态评估报告和摘要
- `POST /api/cmdc/advanced`：运行配对 Bootstrap、缺失模态、嵌套模型、校准和决策曲线分析
- `POST /api/cmdc/holdout`：按固定种子划分受试者级 80/20 留出集，仅用训练集选择阈值并锁定测试集评估；body 可传 `{"repeats": 10, "bootstrap_iterations": 1000}`

## MODMA 辅助语音接口

- `GET /api/modma/summary`：读取 MODMA 授权范围、52 人结构审计、语音基线和 MPDD 迁移结论
- `POST /api/modma/audit`：重新核对受试者目录、1,508 个 WAV、标签和文件头
- `POST /api/modma/train_audio`：提取声学特征并执行 5×5 受试者分组源内基线
- `POST /api/modma/transfer`：在同一组 MPDD 受试者切分上比较目标域基线、零样本、概率堆叠和联合训练
- MODMA 为 18 至 55 岁临床 MDD/HC 辅助数据，接口结果不得用于宣称老年外部验证性能或进入当前主风险评分

### 评估与量表

- `POST /api/assessment`：对话 + 量表 + 人口学/共病 + 表情摘要 -> 抑郁概率；校验消息结构、量表名称与题数、分值范围、年龄、性别、共病、睡眠、自评健康和风险概率范围，并返回量表完成度、有效信号和证据边界
- `POST /api/scales/score`：CESD-10 / PHQ-9 / GDS-15 / HAMD-24 计分；非法量表、超量题目和越界分值返回 400

### 用户记录

- `GET /api/users`：列出用户档案，支持 `?q=关键词` 按姓名/编号/备注查询
- `POST /api/users`：创建用户，body 为 `{"name": "...", "age": 76, "gender": "female", ...}`
- `GET /api/users/<user_id>`：读取单个用户
- `PUT /api/users/<user_id>`：更新用户档案
- `DELETE /api/users/<user_id>`：删除用户及其全部评估记录
- `GET /api/users/<user_id>/assessments`：读取历史评估，支持 `?risk=low|medium|high` 和 `?scale=cesd10|phq9|gds15|hamd24`
- `POST /api/users/<user_id>/assessments`：保存一条评估记录，body 为 `{"record": {...}}`；记录必须包含 `storage_consent=true` 与 `consent_recorded_at`

记录保存在 `data/user_records.sqlite`，账号与登录会话保存在 `data/auth.sqlite`，仅本机使用；两个数据库均已加入 `.gitignore` 并限制为 `0600`。密码和密保答案只保存哈希。`data/账号清单.txt` 仅保存账号元数据，不保存密码。

第三方大模型调用默认关闭，必须显式传入 `cloud_consent=true`；发送前会隐藏姓名、住址、电话、证件号和邮箱。更多数据治理要求见 `docs/19_科研工程伦理与数据治理.md`。

### 模拟数据与数据画像

- `POST /api/synthetic/generate`：生成对话/表格/多模态模拟组
- `GET /api/synthetic/status`：查看模拟组状态
- `POST /api/synthetic/delete`：删除模拟组文件
- `POST /api/real_profile/generate`：生成真实数据特征画像
- `GET /api/real_profile/status`：读取真实数据特征画像
- `POST /api/synthetic/v3/run`：运行层次化多模态模拟生成与对比
- `GET /api/synthetic/v3/status`：读取 v3 验证与对比状态
- `POST /api/synthetic/v3/grid`：运行 v3 参数网格搜索与最优对比
- `POST /api/synthetic/v3/augment`：运行片段级增强实验（真实片段 + 小噪声）
- `POST /api/synthetic/dialogue/validate`：运行对话模拟组验证

### Agent 接入

负责 Agent 的同学只需要做两件事：

1. 阅读 `backend_api/agent_adapter.py`，把 `AgentAdapter.chat()` 的占位逻辑替换为真实 LLM Agent。
2. 保持接口结构不变，前端和后端会自动调用：

```json
POST /api/agent/chat
{
  "session_id": "sess-001",
  "messages": [{"role": "user", "text": "最近睡不好"}],
  "context": {
    "mode": "interview",
    "age": 74,
    "gender": "female",
    "scale_name": "cesd10"
  }
}
```

返回：

```json
{
  "reply": "睡不好确实会让人白天更累...",
  "next_question": "以前让您高兴的事，现在还愿意做吗？",
  "agent": "adaptive-interviewer",
  "mode": "interview",
  "topic": "sleep",
  "risk_cues": ["睡眠差"],
  "cue_confidence": {"睡眠差": 0.9},
  "covered_topics": ["sleep"],
  "safety_alert": false
}
```

完整契约见 `GET /api/agent/schema`。

## Agent 的大模型增强

Agent 支持两种模式：

1. 本地规则访谈：无 Key 时仍可运行，按睡眠、兴趣、情绪、躯体、认知、社交、无望与安全风险动态追问。
2. 大模型增强：支持 OpenAI 及兼容接口，例如 DeepSeek。设置 `OPENAI_API_KEY`、`OPENAI_MODEL` 和可选的 `OPENAI_BASE_URL`（推荐写入项目根目录 `.env`），或在前端侧边栏填写 API Key。LLM 调用失败会自动回退到本地规则访谈。

`.env` 已加入 `.gitignore`，不会把密钥提交到仓库；提交前请使用 `.env.example` 作为模板。

## 数据接入约定

后期拿到正式数据时，直接替换/生成以下文件即可，无需改前端：

- `data/charls_standardized.csv`：CHARLS 标准表
- `data/mpdd_features.csv`：MPDD 合并特征表
- `data/mpdd_features_uploaded.csv`：前端上传的 MPDD 临时文件
- `data/cmdc_features.csv`：CMDC 去标识化受试者级特征表，权限 `0600`
- `data/modma/`：MODMA 授权语音与元数据，仅限本机受限目录，不进入 Git

新 Agent 只需要通过 `/api/agent/chat` 交互，不需要直接读取数据库或模型文件。

## 融合权重训练

最终风险概率默认使用文献先验权重；拿到“信号概率 + 标签”的训练数据后，可以训练逻辑回归融合模型并自动切换：

```csv
text_risk,scale_risk,context_risk,expression_risk,mpdd_risk,depression
0.62,0.40,0.55,0.70,0.60,1
0.30,0.22,0.38,0.45,,0
```

- `mpdd_risk` 可留空；系统分别训练“无 MPDD”和“含 MPDD”两个融合模型。
- 训练数据上传到工作台“融合校准”页，或调用 `POST /api/fusion/train`。
- 模型保存在 `output/fusion_models/`，未训练或文件损坏时自动回退到文献先验权重。

## 模型产物

训练后模型会保存到：

- `output/tabular_models/`：Logistic、RandomForest、XGBoost 及特征重要性
- `output/multimodal_models/`：多模态融合模型
- `output/cmdc_models/`：CMDC 独立外部验证模型，不参与老年主风险评分
- `output/cmdc_locked_holdout_model.pkl`：固定受试者留出分析的锁定模型，不参与老年主风险评分
- `output/modma/`：MODMA 结构审计、声学特征和源内基线
- `output/modma_transfer/`：MODMA→MPDD 目标域迁移报告，不进入主风险评分
- `output/fusion_models/`：学习型风险融合权重与报告

通过 `joblib.load()` 即可直接复用，无需重新训练。
