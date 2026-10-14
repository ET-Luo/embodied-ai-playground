# Dataset adapters and inspection metadata

`rlds_toy.py` 只支持 `synthetic_loaded_rlds_xy/1.0`：Python/JSON 模拟canonical loaded steps。
不是TFDS reader，不能输入Fractal/Bridge的真实dict后猜测单位或语义。
`oxe_schema_notes.json` 是两份官方TFDS目录的人工字段摘要，标注URL/version/date，非真实episode。
详见 [S0.3学习包](../../docs/vla/s0_3_rlds_oxe_concepts.md)。

完整source reader未来停留在边界：TFDS decoded episodes → source-specific audit/mapping → local NPZ+JSON。
本toy的约4KB source由实验生成到ignored tmp，不下载或安装大型依赖。
从repo根目录执行 `scripts/s0_3_rlds_concepts.py`，adapter直接import本repo S0.1 validator。
不导入sibling repo，不依赖RLDS框架；奖励/discount未进入本地v1，以metadata记录丢弃。
缺source units/dt/有效终点/success或恒定language时应拒绝、隔离或显式扩展契约。
真实数据adapter需逐dataset验证，不能把toy profile当OXE通用标准。

## S0.4 source snapshot

`bridge_sample_manifest.json`固定社区Bridge LeRobot mirror的一条episode、一个camera及meta/task文件。
它是inspection provenance，不是转换profile；未知单位/frame/controller/producer与终点/success不可据此补齐。
脚本`../../scripts/s0_4_inspect_robot_sample.py`做bounded download、hash与结构检查，保存report/预览到ignored tmp。
四文件约1.93MB；原始数据只读保留，不接入S0.1/S0.2训练格式。
