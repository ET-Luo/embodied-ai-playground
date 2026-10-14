# Dataset boundary

S0.4已下载一条公开Bridge社区LeRobot转存及单相机到ignored `tmp/`，原始RLDS尚未读取。
S0.1 的三步 toy episode 由脚本生成到 ignored `tmp/`。
`toy/` 留作未来小数据，`adapters/` 包含S0.3合成RLDS概念适配器；S0.2 的 state-only sequence dataset 已实现于 [脚本](../scripts/s0_2_sequence_dataset.py)。

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

## S0.3 schema boundary

[adapter notes](adapters/README.md)：只有explicit synthetic loaded-step profile，不是通用RLDS reader。
[oxe_schema_notes.json](adapters/oxe_schema_notes.json) 是官方Fractal/Bridge catalog字段的人工摘要，非真实episode。
`scripts/s0_3_rlds_concepts.py` 生成约4KB JSON fixture到ignored tmp，转为本地NPZ+JSON。
S0.3当时未安装TensorFlow/TFDS/RLDS、未下载真实robot dataset；后续S0.4检查见下文。

## S0.4 real sample inspection boundary

[学习包](../docs/vla/s0_4_real_robot_dataset_inspection.md)、[manifest](adapters/bridge_sample_manifest.json)。
四份固定文件1,923,948bytes，SHA256校验；28行state/action和28帧单相机视频。
raw保留原样，HTML/PNG/JSON是派生inspection，不是本地训练episode。
source转存代码、原始episode/reset group、action units/frame/controller、success与boundary未明确；不猜测转换。
PyArrow/av/Pillow仅为读取和预览，未使用TensorFlow/TFDS、未训练模型或执行机器人命令。

## S0.5 raw action and derived views

[学习包](../docs/vla/s0_5_continuous_action_representation.md)：four-command joint_absolute servo toy，四独立resets，先固定train3/validation1。
raw NPZ+JSON仍为embodied_episode/1.0，rad references/measurements、T/T+1、显式controller与boundary/success。
derived joint/Cartesian views、各自normalizer和chunk存独立ignored tmp artifacts；不覆盖raw或读取真实Bridge样本。
统计只fit原始train actions各一次；validation、padding、重叠window复制不参与；无clip，constant channel政策显式。
