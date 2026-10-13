# Dataset boundary

当前没有下载公开 robot dataset。S0.1 的三步 toy episode 由脚本生成到 ignored `tmp/`。
`toy/` 留作未来小数据，`adapters/` 留作文件适配器；S0.2 的 sequence dataset 尚未实现。

每个 episode 是同名 `.npz` + `.json`，见 [contract](../docs/data_contract.md)。
NPZ 只含数值/布尔数组，读取必须 `allow_pickle=False`；文本和语义放 JSON。
原始数据保持物理单位，normalizer 是独立的 train-only artifact。
未来导入旧项目时记录源 schema、commit、episode/reset group、controller 和 sampling rate。
保留失败轨迹；success 不等于每步标签质量。按 episode / reset group 分割，不能随机拆 timestep。
公开数据下载前说明大小、用途、必要性；不自动下载 GB/TB 级数据。
