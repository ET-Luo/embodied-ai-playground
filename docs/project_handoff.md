# P4 project handoff

Updated 2026-10-13 (Europe/Berlin)。新会话读 AGENTS.md → 本文 → README → 当前学习包。

Current Task = **S0.1 — VLA Data Contract**。Engineering Complete + Learning Mastered（2026-10-13 用户确认）。
Next Task = **S0.2 — Episode / Sequence Dataset**，仅计划，不授权自动进入。
当前没有 Tiny VLA、OpenVLA、World Model 实现；后续阶段只有 roadmap 与空目录。

## Initialization and inherited experience

初始 git status --short 为空，HEAD `89822df Initial commit`；只有 README、MIT LICENSE、
GitHub Python .gitignore。保留 LICENSE 和已有 ignore，扩展 README 与本地 artifact ignore。
创建用户要求的目录骨架、最小 requirements、长期规则、Stage 0–4 roadmap、数据契约与 S0.1 包。

Sibling ../mujoco-robotics-playground 存在；只读检查 AGENTS、README、handoff、Stage19–22 文件结构，
并查看 S19.1 episode_contract、S22.2 recorder、S22.6 dexterous_bc 源码。
旧交接记录 P3 截止 S22.9，Engineering + Learning 完成（历史用户确认，未重新运行旧实验）。
继承 Sprint Learning、双状态、T/T+1、controller/action 语义、train-only normalization、
episode/reset-group split、全分母失败保留、因果 force/tactile、offline loss 与 closed-loop 成功分开、
冻结 benchmark/OOD/latency 评价。不继承旧 repo 内部 import、MuJoCo 环境耦合或旧规则作用域。
S22.6 nominal expert 可成功但 BC 仍失败，提醒不能用低 loss 推断部署能力。
候选导出接口和映射见 [data contract](data_contract.md)；本轮未实现 exporter 或读取旧数据。
旧仓库未修改、未复制、未做 submodule，未执行旧 Python。

## S0.1 artifacts and validation

[Learning package](vla/s0_1_data_contract.md)，[script](../scripts/s0_1_data_contract.py)。
NPZ + JSON `embodied_episode/1.0`；3 actions、4 observations、2D world displacement、10Hz。
语言是 string，标签是手写 deterministic toy；没有 learned model。
Validator 支持必需字段和显式 optional next_state/image/force/tactile，拒绝未知 schema/数组。
只证明结构和声明可检查，不能自动证明实际 sensor causality/frame 或物理正确性。

本机：WSL Linux，`.venv/bin/python`，Python 3.14.7 / NumPy 2.5.4；
NumPy 实际模块位于 `.venv/lib/python3.14/site-packages/numpy/__init__.py`。
初始默认 Python 为 Conda base 且无 NumPy；系统 Python3.12 venv 因缺 ensurepip 失败。
使用现有 `/home/lucas/miniconda3/bin/python -m venv --clear .venv` 创建隔离环境，
`.venv/bin/python -m pip install 'numpy>=1.26,<3'`，下载 wheel 约16.7MB；未向 base/system 安装。
未来 ML 优先 Python3.11/3.12，后续 Torch 兼容尚未验证；本次 NumPy 能运行。

Executed:

```bash
.venv/bin/python scripts/s0_1_data_contract.py
.venv/bin/python scripts/s0_1_data_contract.py --direction left --output tmp/s0_1_left
.venv/bin/python -m pip check
```

左右两个方向均 PASS round-trip / toy transition alignment / same-state different-language labels，
各 8/8 invalid cases 被拒绝；终点 ±0.029999999329447746m；pip check 无冲突。
独立 inline check 验证 optional next_state/image/force/tactile 合法与 next_state 错配拒绝。
Markdown 本地链接检查与 git diff --check 通过；输出 ignored `tmp/`，环境 ignored `.venv/`。
未下载公开 robot dataset 或模型权重；没有图像训练、sequence/padding/split 实现。

## Learner handoff

Run：`.venv/bin/python scripts/s0_1_data_contract.py --output tmp/my_s0_1`，读 NPZ/JSON。
Modify：先预测 left action/terminal/success，再用 `--direction left --output tmp/my_s0_1_left` 核对。
Explain：回答学习包五问，画三转移四观测时序并区分真实 servo reference 与 toy delta。
只在用户明确确认后更新 README Learning boxes。完成后等待选择 S0.2，不自动推进。

## S0.1 learning confirmation — 2026-10-13

用户明确确认实验和预测均已完成，五项 Explain 准确覆盖 T 次转移/T+1 观测、
absolute reference 与 measured-state delta、float64 时钟/float32 向量、
success/next_state 的未来信息泄漏、同 shape 不等于同 embodiment 语义。
README 与学习包 Run/Modify/Explain 已全部勾选。这里只同步学习状态，
未重新执行 runtime；工程结果沿用上述 2026-10-13 验证。
Current Task 仍为 S0.1（已掌握），Next Task 为 S0.2；等待用户选择，不自动实现。
