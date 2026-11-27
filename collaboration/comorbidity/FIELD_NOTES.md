# 共病轨迹字段说明

## 推荐长表

- `subject_id`：去标识化受试者编号，与主表保持稳定对应。
- `disease_code`：统一疾病代码，建议使用 `hibpe`、`arthre`、`dyslipe` 等。
- `diagnosed`：`1` 表示曾被医生诊断，`0` 表示明确未诊断，空白表示未知。
- `onset_age`：首次确诊年龄；未知时留空，不得用当前年龄代替。
- `onset_year`：首次确诊年份；与 `onset_age` 至少提供一个。
- `source_wave`：信息来源波次，例如 `2011`、`2013`、`2015`、`2018`。
- `confirmed_by`：`self_report`、`medical_record`、`doctor` 或 `unknown`。

## 宽表兼容

如果只能提供宽表，每个病种至少提供：

```text
chronic_<disease>
chronic_<disease>_onset_age
```

其中第二个字段可以为空。项目会区分“精确发病年龄轨迹”和
“波次观察轨迹”，不能把后者写成精确临床发病时间。

## 禁止推断

- 不根据病种常识补写发病顺序。
- 不把同一次调查中同时报告的病种强行排序。
- 不使用未来波次诊断回填早期波次。
- 不把 `chronic_psyche` 直接等同于抑郁症诊断。
