# Dataset boundary

当前没有下载公开 robot dataset。S0.1 的三步 toy episode 由脚本生成到 ignored `tmp/`。
`toy/` 留作未来小数据，`adapters/` 留作文件适配器；S0.2 的 state-only sequence dataset 已实现于 [脚本](../scripts/s0_2_sequence_dataset.py)。

每个 episode 是同名 `.npz` + `.json`，见 [contract](../docs/data_contract.md)。
NPZ 只含数值/布尔数组，读取必须 `allow_pickle=False`；文本和语义放 JSON。
原始数据保持物理单位，normalizer 是独立的 train-only artifact。
未来导入旧项目时记录源 schema、commit、episode/reset group、controller 和 sampling rate。
保留失败轨迹；success 不等于每步标签质量。按 episode / reset group 分割，不能随机拆 timestep。
公开数据下载前说明大小、用途、必要性；不自动下载 GB/TB 级数据。

## S0.2 derived views

脚本生成12个 variable-length toy episodes（6 reset groups，各有左右两条），写入 `tmp/<run>/episodes/`。
读取并验证 raw 文件后按 group split，再生成 train/validation windows。
`*_windows.npz` 含 observation/action、bool masks、source indices、anchor 与 Unicode ID/language；
`report.json` 保存 seed、L/H、split IDs、shape/count、诊断结果。所有产物 ignored。
数值保持 world m，padding=0但有效性只由mask决定；未训练模型，未实现完整normalizer。
该派生格式不是raw episode schema升级。详见 [学习包](../docs/vla/s0_2_sequence_dataset.md)。
