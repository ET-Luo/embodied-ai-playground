# P4 — Embodied Foundation Models: VLA & World Models

CPU-first educational and experimental playground for understanding VLA systems,
robot foundation policies, action representations, world models, and embodied decision-making.
Project-driven · experiment-backed · interview-explainable · hardware-aware.
本项目学习 foundation-model 层，不声称训练了大型 foundation model。

`mujoco-robotics-playground` 提供 P0–P3 传统机器人与 Robot Learning 基础；
本仓库独立，通过 versioned dataset / file contract 连接，不导入旧仓库模块。
不重复 MuJoCo、运动学、规划、ROS2、接触控制、BC / ACT / Diffusion 基础。

```mermaid
flowchart TD
  O[Vision + Language + Proprioception] --> V[VLA / Foundation Policy]
  V --> A[Action / Chunk]
  A --> E[Environment]
  E --> N[New Observation]
  N --> W[World Model]
  W --> F[Predicted Future / Success / Risk]
  F --> R[Replan / Execute]
  R --> E
```

以上是最终能力链；当前完成Stage0及S1.1–S1.3工程包：数据契约、sequence dataset、RLDS/OXE概念与真实样本检查、continuous/discrete action与RGB-language chunk toy实验。已实现literal语言条件state MLP、tiny learned word embedding和tiny CNN定位；未实现learned VLA或world model。

| Stage | 主线 | 状态 |
|---|---|---|
| 0 | Embodied AI Data & Action Representation | S0.1–S0.7 Engineering Complete + Learning Mastered |
| 1 | Tiny Vision-Language-Action Policy | S1.1–S1.2 Engineering Complete + Learning Mastered；S1.3 Engineering Complete，Learning待确认 |
| 2 | Modern VLA Architecture Study | Planned |
| 3 | Robot World Models | Planned |
| 4 | VLA × World Model Integration | Planned |

Current Task = **S1.3 — Tiny Visual Encoder**
Next Task = **S1.4 — Multimodal Fusion**（仅计划）

S0.1 Engineering:
- [x] Code — toy episode、NPZ + JSON、validator
- [x] Experiment — CPU round-trip / alignment / rejection checks
- [x] Docs — contract 与 learning package

S0.1 Learning:
- [x] Run
- [x] Modify
- [x] Explain

Engineering Complete + Learning Mastered（2026-10-13 用户确认实验、预测及五项 Explain）。Sprint Learning Mode：Codex 完成当前任务工程包，用户 Run → Modify → Explain，然后决定是否进入下一任务。

## S1.3 status and Run

Engineering:

- [x] Code — 116参数NumPy CNN、15×15×4空间features、heatmap与显式pixel/world映射
- [x] Experiment — 未见目标中心定位、GAP/颜色probe、oracle与visible/occluded/blank完整闭环
- [x] Docs — [S1.3学习包](docs/vla/s1_3_tiny_visual_encoder.md)

Learning:

- [ ] Run
- [ ] Modify
- [ ] Explain

CPU NumPy，无安装/下载；Python3.14.7 / NumPy2.5.4。PNG使用标准库，不依赖当前缺失的Pillow。
先读学习包，再运行；第二条先预测主条件切换的不变量和失败：

```bash
.venv/bin/python scripts/s1_3_visual_encoder.py --output tmp/my_s1_3
.venv/bin/python scripts/s1_3_visual_encoder.py --observation occluded --output tmp/my_s1_3_occluded
```

每次都保存三条件对照，flag只选主报告条件，不改变训练。validation24个未见中心，visible平均定位误差.000309m，
toy闭环24/24；occluded/blank均0/24。查看preview.png（三列RGB/相对热图/遮红）、features与report。
只证明固定颜色格点renderer的CNN定位+固定控制器；不是learned VLA或真实视觉精度。S1.4仅计划。

## S1.2 status and Run

Engineering:

- [x] Code — train-only tokenizer、4维learned embedding、masked mean、310参数state policy
- [x] Experiment — embedding反向核验、PAD/UNK/词序诊断、held-out闭环与完整OOV失败
- [x] Docs — [S1.2学习包](docs/vla/s1_2_tiny_language_representation.md)

Learning:

- [x] Run
- [x] Modify
- [x] Explain

CPU NumPy，无安装/下载；repo .venv Python3.14.7 / NumPy2.5.4。
先读学习包Problem→Mathematics，第二条先预测reject→map的影响：

```bash
.venv/bin/python scripts/s1_2_language_representation.py --output tmp/my_s1_2
.venv/bin/python scripts/s1_2_language_representation.py --unk-policy map --output tmp/my_s1_2_map
```

两模式权重和known结果一致：validation raw MSE≈1.607e−6m²、toy success12/12。
OOV两模式均0/24：默认全部tokenization拒绝，map则全部在10步终点失败。
查看tokenizer.json、report diagnostics及train/validation_tokens.npz；PAD不计入mean，零UNK仍计入。
仅固定两指令toy证据，不是自然语言泛化。Engineering Complete + Learning Mastered（2026-10-17 用户确认实验、预测及五项 Explain）；S1.3仅计划。

## S1.1 status and Run

Engineering:

- [x] Code — 226参数NumPy MLP、手写反向/Adam、literal one-hot与无语言基线
- [x] Experiment — reset-group held-out、梯度核验、物理MSE、完整闭环/失败/延迟
- [x] Docs — [S1.1学习包](docs/vla/s1_1_language_conditioned_state_policy.md)

Learning:

- [x] Run
- [x] Modify
- [x] Explain

CPU，无新依赖或下载；实测repo .venv Python3.14.7 / NumPy2.5.4。
先读学习包Problem→Mathematics，再运行；第二条先预测训练步数的影响：

```bash
.venv/bin/python scripts/s1_1_language_state_policy.py --output tmp/my_s1_1
.venv/bin/python scripts/s1_1_language_state_policy.py --epochs 20 --output tmp/my_s1_1_short
```

默认含语言/无语言validation raw MSE≈1.349e−6/4.669e−4 m²，toy闭环12/12与6/12；
20步为3/12与1/12。只证明固定literal两目标的理想点任务，不是视觉语言泛化。
Engineering Complete + Learning Mastered（2026-10-17 用户确认实验、预测及五项 Explain）。
当前S1.1已掌握；S1.2仅计划，等待用户选择。

## S0.7 status and Run

Engineering:

- [x] Code — causal state + anchor RGB + language → rule chunks、S0.2 mask/index audit、execution prefix
- [x] Experiment — paired language、future/result isolation、12 rejection cases、H/E与扰动闭环toy比较
- [x] Docs — [S0.7 learning package](docs/vla/s0_7_action_chunk_representation.md)、context preview与完整失败记录

Learning:

- [x] Run
- [x] Modify
- [x] Explain

Engineering Complete + Learning Mastered（2026-10-14 用户确认实验、预测及五项 Explain）。已有NumPy/Pillow，无安装或下载。
6 synthetic point episodes，32×32 RGB、world delta[m]、10Hz；手写规则，不是learned VLA。
output必须为新目录；先预测H4固定、E1→4的变化，再运行第二条：

```bash
.venv/bin/python scripts/s0_7_action_chunk.py --output tmp/my_s0_7
.venv/bin/python scripts/s0_7_action_chunk.py --execute 4 --output tmp/my_s0_7_e4
```

查看context_preview.png、report.json及train/validation_chunks.npz；train16/validation8 anchors、42/20 valid tokens。
离线masked MSE均0，nominal toy均2/2；固定扰动E1为1/2、E4为0/2，完整失败保留。
H是未来输出长度，E是执行前缀；mask不传入planner。Stage0 Engineering Complete + Learning Mastered，等待用户选择S1.1。

## S0.6 status and Run

Engineering:

- [x] Code — train-range scalar bins / channel IDs / center decode / explicit saturation
- [x] Experiment — B8→16、boundary/half-bin oracle、16 rejection cases、physical error与toy replay
- [x] Docs — [S0.6 learning package](docs/vla/s0_6_discrete_action_tokenization.md)、codec与raw语义边界

Learning:

- [x] Run
- [x] Modify
- [x] Explain

Engineering Complete + Learning Mastered（2026-10-14 用户确认实验、预测及五项 Explain）。已有NumPy，无安装或下载。
复用S0.5已知joint_absolute toy；range只fit12条原始train actions；decode有损，held-out saturation显式报告。
output必须为新目录；第二条先预测B8→16再运行：

```bash
.venv/bin/python scripts/s0_6_action_tokenization.py --output tmp/my_s0_6
.venv/bin/python scripts/s0_6_action_tokenization.py --bins 16 --output tmp/my_s0_6_b16
```

查看tokenizer.json、tokens.npz与report.json。词表16→32，每episode仍8IDs；half-bin界减半，held-out仍5/8饱和。
toy replay以原reference为目标，不能把低重建误差叫真实robot success；未训练模型或复现大型tokenizer。
完成Run/Modify/五项Explain后等待选择S0.7。

## S0.5 status and Run

Engineering:

- [x] Code — joint absolute / two delta anchors / Cartesian views、train-only normalizers、S0.2 chunks
- [x] Experiment — correct decode/replay、gain .5→1、frame/embodiment/leakage、11 rejection cases、H1/H6
- [x] Docs — [S0.5 learning package](docs/vla/s0_5_continuous_action_representation.md)、physical/raw与derived语义

Learning:

- [x] Run
- [x] Modify
- [x] Explain

Engineering Complete + Learning Mastered（2026-10-14 用户确认实验、预测及五项 Explain）。只使用已有NumPy，无下载或安装。
四条手写servo toy episodes（train3 / validation1），不是Bridge转换或learned policy。
output必须为新目录；先预测gain变化，再运行第二条：

```bash
.venv/bin/python scripts/s0_5_action_representation.py --output tmp/my_s0_5
.venv/bin/python scripts/s0_5_action_representation.py --gain 1 --output tmp/my_s0_5_gain1
```

查看report.json、raw episodes、representations.npz、normalizers.json和train_chunks.npz。
gain=.5时reference≠下一测量，两种delta须分别解码；normalization只fit原始train actions，padding不参与。
Cartesian views仅为XY目标，无IK/controller执行转换。完成Run/Modify/五项Explain后等待选择S0.6。

## S0.4 status and Run

Engineering:

- [x] Code — pinned bounded download、Parquet/schema audit、CPU video decode、HTML/PNG/report
- [x] Experiment — real episode0、28 rows/frames、clock/PTS、8 rejection cases、preview stride invariance
- [x] Docs — [S0.4 learning package](docs/vla/s0_4_real_robot_dataset_inspection.md)、manifest与契约差异

Learning:

- [x] Run
- [x] Modify
- [x] Explain

Engineering Complete + Learning Mastered（2026-10-14 用户确认实验、预测及五项 Explain）。
首次下载四文件共1,923,948bytes（约1.93MB）：社区BridgeData V2 LeRobot转存的一条episode、
单相机视频和schema/task索引；不是原始RLDS。NumPy/PyArrow/av/Pillow，CPU；不下载全集或模型。
本机隔离环境现为Python3.12.3；输出目录必须不存在：

```bash
.venv/bin/python scripts/s0_4_inspect_robot_sample.py --download --output tmp/my_s0_4
.venv/bin/python scripts/s0_4_inspect_robot_sample.py --preview-stride 14 --output tmp/my_s0_4_stride14
```

打开`tmp/my_s0_4/inspection.html`或PNG/report.json；第二条运行前预测5→3预览帧及raw/time/shape是否改变。
现有source只核验不覆盖；无`--download`时离线。缺终点/边界/success与控制语义证据，拒绝训练契约导出。
完成Run/Modify/五项Explain后等待选择S0.5。

## S0.3 status and Run

Engineering:

- [x] Code — explicit synthetic loaded-RLDS adapter / local contract validation
- [x] Experiment — last action invalid、terminal/truncated、independent success、10 rejection cases
- [x] Docs — [S0.3 learning package](docs/vla/s0_3_rlds_oxe_concepts.md)、two official catalog schema notes

Learning:

- [x] Run
- [x] Modify
- [x] Explain

Engineering Complete + Learning Mastered（2026-10-14 用户确认实验、预测及五项 Explain）。
只使用已有NumPy；没有TensorFlow/TFDS/RLDS安装或真实数据下载。输出目录必须不存在：

```bash
.venv/bin/python scripts/s0_3_rlds_concepts.py --output tmp/my_s0_3
.venv/bin/python scripts/s0_3_rlds_concepts.py --ending truncated --output tmp/my_s0_3_truncated
```

Run后看source四steps与本地三actions；Modify前预测boundary/success/shape变化；Explain回答学习包五问。
本例是合成概念验证，未声称完整OXE转换或真实机器人能力。完成后等待选择S0.4。

## S0.2 status and Run

Engineering:

- [x] Code — file-backed episodes、group split、causal windows、padding/masks
- [x] Experiment — source alignment、masked loss、boundary/leakage counterexamples
- [x] Docs — [S0.2 learning package](docs/vla/s0_2_sequence_dataset.md)

Learning:

- [x] Run
- [x] Modify
- [x] Explain

Engineering Complete + Learning Mastered（2026-10-14 用户确认实验、预测及五项 Explain）。
只需要已有 NumPy，不增加依赖。仓库根目录执行，输出目录须不存在：

```bash
.venv/bin/python scripts/s0_2_sequence_dataset.py --output tmp/my_s0_2
.venv/bin/python scripts/s0_2_sequence_dataset.py --horizon 2 --output tmp/my_s0_2_h2
```

第一条为 Run；第二条在预测 horizon 变化后做 Modify。阅读学习包回答五项 Explain。
完成后等待用户选择下一任务。

## Run S0.1

Python 3.11+；建议未来 ML 任务使用 3.11/3.12。已有本地 `.venv` 可直接使用。
首次设置（选择支持 venv 的 Python，不向 base/system 安装）：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/s0_1_data_contract.py
```

默认输出 `tmp/s0_1/`，已有输出不会覆盖；再次运行请选新目录：

```bash
.venv/bin/python scripts/s0_1_data_contract.py --output tmp/my_s0_1
```

阅读 [S0.1 learning package](docs/vla/s0_1_data_contract.md)、[data contract](docs/data_contract.md)、[roadmap](docs/roadmap.md)、[handoff](docs/project_handoff.md)、[data policy](data/README.md)。

## Scope and layout

当前依赖为NumPy与S0.4读取/预览所需PyArrow、av、Pillow。后续需要时才添加CPU PyTorch / Matplotlib。
32×32 / 64×64 图像、小模型、小数据、低维状态与小规模闭环是默认预算。
完整版模型在 docs 中学习论文、代码与 tensor trace，再设计 Tiny / CPU Learning Version。
当前非目标：7B VLA 训练、full OpenVLA / π0 复现或 fine-tuning、大规模 LoRA、TB 数据、
大型视频生成、humanoid foundation model、多 GPU、Isaac Sim/Lab、LLM Agent framework 堆叠。

`docs/` 学习包与交接；`data/` 文件边界与未来 adapters；`environments/manipulation/` 环境；
`vla/` Tiny VLA、动作表示、架构追踪；`world_models/` state/latent dynamics 与 planning；
`benchmarks/` 评价；`scripts/` 当前实验。空目录是骨架，不代表已实现。
