# 数据说明

## CHARLS

脚本 `scripts/01_build_charls_dataset.py` 会读取本机已有的 Harmonized CHARLS 数据，并在 `data/charls_standardized.csv` 输出标准表格。

## MPDD

MPDD 数据经官方 MPDD Dataset License Agreement 申请获得，仅限学术研究使用，禁止商业用途与任何形式的再分发。官方申请入口与完整协议见 `docs/07_MPDD数据使用合规.md`。

已获取的官方特征放在 `data/mpdd/`，目录结构为：

```text
data/mpdd/
  Train-MPDD-Elder/
    Elder/
      Audio/train/{mfcc, opensmile, wav2vec2}/<subject>/*.npy
      Video/train/{densenet, resnet, openface}/<subject>/*.npy
      IMU/train/<subject>/*.npy
      split_labels_train.csv
      descriptions_embeddings_with_ids.npy
  Test-MPDD-Elder (1)/...
```

然后运行：

```bash
python scripts/02_build_mpdd_dataset.py --mpdd-root data/mpdd
```

脚本会按被试 ID 聚合各模态的时间序列特征（均值、标准差、最小值和最大值），
并把 PHQ-9 同时写入 `phq9_score` 和 `hamd_score`，输出到 `data/mpdd_features.csv`。
工作台“多模态抑郁检测”页也可以直接点击“从本地 MPDD 原始特征构建数据表”。

另一个入口是 utterance 级特征表 `data/mpdd_utterance_features.csv`：
每个 `A_*.npy / V_*.npy / IMU *.npy` 单独占一行，按被试合并标签和人格嵌入。
该表用于更稳定的 subject-aware 基线：

```bash
python scripts/04b_train_mpdd_subject_baseline.py
```

输出 `output/multimodal_subject_report.csv`（重复交叉验证汇总）和
`output/subject_fusion_pipeline.pkl`（可用于新被试推理的完整管线）。

进阶评估：

```bash
python scripts/04c_train_mpdd_advanced.py
```

会生成 label3 三分类报告 `output/multimodal_subject_label3_report.csv` 和
PHQ-9 回归报告 `output/multimodal_subject_phq9_score_report.csv`。
测试集预测可导出到 `output/mpdd_test_predictions.csv`。

## CMDC

CMDC 数据来自 IEEE DataPort，使用前需遵守项目获得的数据 EULA。原始加密压缩包默认放在项目上级的 `新数据集1/`，不复制进项目；归档密码只通过环境变量或脚本交互输入，不写入代码、日志、数据库或派生表。

构建去标识化受试者级特征表：

```bash
export CMDC_ARCHIVE_PASSWORD="由数据方提供的密码"
python scripts/50_build_cmdc_features.py
```

生成文件：

- `data/cmdc_features.csv`：78 名受试者的数值特征，权限 `0600`
- `output/cmdc_data_audit.json`：模态覆盖、受试者数、合规边界

训练 CMDC 外部验证模型：

```bash
python scripts/51_train_cmdc_models.py --repeats 10
```

生成 `output/cmdc_benchmark_report.csv`、`output/cmdc_benchmark_summary.json` 和独立模型。CMDC 是混合成人样本，不是老年专属队列，不进入当前老年风险主评分。

## MODMA

MODMA 已由导师完成官方申请并确认可用于本项目，定位为跨年龄段中文临床抑郁辅助数据。原始语音、EEG 和受试者量表统一放在 `data/modma/`，不进入 Git 仓库或公开共享渠道。

当前阶段只接入语音：

```text
data/modma/
  metadata/
    MODMA_dataset_description.pdf
    subjects_information_audio_lanzhou_2015.xlsx
  <subject_id>/
    1.wav ... 29.wav
```

运行受试者级数据审计：

```bash
PYTHONPATH=src /opt/anaconda3/bin/python scripts/64_audit_modma.py
```

输出：

- `output/modma/modma_subject_index.csv`
- `output/modma/modma_audio_audit.json`

MODMA 受试者为 18 至 55 岁临床 MDD/健康对照，不是老年专属数据集，不能与 CHARLS、MPDD 或 CMDC 拼接为同一个人，也不能用于宣称老年外部验证性能。完整边界见 `docs/24_MODMA辅助数据接入与使用边界.md`。

当前语音审计状态：52 人、1,508 个 WAV、约 3.2 GB；52 人全部匹配元数据，1,503 个 WAV 可读取。受试者 `2010004` 的第 24 至 28 段源文件不是有效 RIFF/WAVE，暂时只将该受试者标记为部分可用，不静默替换或删除异常文件。

首轮源内基线使用 52 名受试者、1,503 段可读语音，按受试者分组执行 5 折重复 5 次：AUC `0.667 ± 0.156`、PR-AUC `0.706 ± 0.155`、平衡准确率 `0.648 ± 0.125`、Brier `0.232 ± 0.060`、ECE `0.269 ± 0.082`。该结果说明数据中有语音信号，但尚不足以直接接入老年风险评分。
