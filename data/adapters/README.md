# S0.3 adapters and metadata

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
