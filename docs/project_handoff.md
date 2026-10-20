# P4 project handoff

Updated 2026-10-20 (Europe/Berlin)。新会话读 AGENTS.md → 本文 → README → 当前学习包。

Current Task = **S4.6 — Final Benchmark**，Engineering Complete，Learning pending。
S4.5 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S4.4 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S4.3 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S4.2 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S4.1 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S3.8 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S3.7 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
Stage3 S3.1–S3.8全部Engineering Complete + Learning Mastered；不代表大视频模型复现或真实机器人能力。
S3.6 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S3.5 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S3.4 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S3.3 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S3.2 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S3.1 Engineering Complete + Learning Mastered（2026-10-20 用户确认）。
S2.7 Engineering Complete + Learning Mastered（2026-10-19 用户确认）。
Stage2 S2.1–S2.7全部工程与学习完成；不代表完整模型复现或机器人控制成功。
S2.6 Engineering Complete + Learning Mastered（2026-10-19 用户确认）。
S2.5 Engineering Complete + Learning Mastered（2026-10-18 用户确认）。
S2.4 Engineering Complete + Learning Mastered（2026-10-18 用户确认）。
S2.3 Engineering Complete + Learning Mastered（2026-10-18 用户确认）。
S2.2 Engineering Complete + Learning Mastered（2026-10-18 用户确认）。
S2.1 Engineering Complete + Learning Mastered（2026-10-18 用户确认）。
S1.7 Engineering Complete + Learning Mastered（2026-10-18 用户确认；策略仍失败）。
S1.6 Engineering Complete + Learning Mastered（2026-10-18 用户确认）。
S1.5 Engineering Complete（闭环仍失败）+ Learning Mastered（2026-10-18 用户确认）。
S1.4 Engineering Complete + Learning Mastered（2026-10-17 用户确认）。
S1.3 Engineering Complete + Learning Mastered（2026-10-17 用户确认）。
S1.2 Engineering Complete + Learning Mastered（2026-10-17 用户确认）。
S1.1 Engineering Complete + Learning Mastered（2026-10-17 用户确认）。
Stage0 S0.1–S0.7 Engineering Complete + Learning Mastered（历史用户确认）。
Next Task = **无自动后续任务**，等待S4.6学习确认与用户选择。
已实现literal条件state MLP、tiny learned word embedding、独立tiny CNN定位和summary融合对照；已有tiny averaged-patch Transformer action chunk但未学成控制；另有独立tiny语言目标泛化诊断和冻结checkpoint闭环评价，未修复S1.5；没有运行真实OpenVLA；已有S3.1–S3.7合成XY模型/规划/反馈及S3.8固定视频world model架构/解析反例学习包，但未运行真实机器人或大型视频World Model；S2.3仅架构与接口学习。

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
S0.1 当时未下载公开 robot dataset 或模型权重；sequence/padding/split 见下面新增 S0.2 记录。

## S0.1 learner handoff (historical)

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
当时 Current Task 为 S0.1（已掌握），Next Task 为 S0.2；2026-10-14 用户请求继续推进后实现 S0.2。

## S0.2 Engineering — 2026-10-14

用户请求“继续推进任务”，只推进下一个小任务 S0.2。
启动 pwd 正确，git status --short 为空；读取规则/README/交接/S0.1 package和源码。
[Learning package](vla/s0_2_sequence_dataset.md)，[script](../scripts/s0_2_sequence_dataset.py)。
requirements 仍只有 NumPy（仅更新覆盖任务注释），无下载/安装。
执行前重新确认 `.venv/bin/python`，Python3.14.7、NumPy2.5.4，实际模块位于 `.venv/lib/python3.14/site-packages/numpy/__init__.py`。
该环境通过本次CPU NumPy实验；未来Torch兼容未验证。

数据：6 independent reset groups，每组左右episode，T1…6，共12episodes/42actions。
写 raw NPZ+JSON，重新读取并调用 S0.1 validator；同reset两episode整组分到train/validation。
先split后window；每个action一个anchor；左history/右action zero padding，True=真实token。
输出带 Unicode IDs/language、indices、masks；allow_pickle=False round-trip通过。
本实现只支持state+language，遇可选传感模态明确拒绝；未训练模型/实现完整normalizer。

Executed:

```bash
.venv/bin/python scripts/s0_2_sequence_dataset.py --output tmp/s0_2_verified
.venv/bin/python scripts/s0_2_sequence_dataset.py --horizon 2 --output tmp/s0_2_h2
.venv/bin/python scripts/s0_2_sequence_dataset.py --history 1 --horizon 1 --seed 3 --output tmp/s0_2_min
.venv/bin/python -m pip check
```

默认L3/H4/seed2：train8episodes/22windows，validation4episodes/20windows；
valid action tokens48/56，history44/48；H2保持split/N，action tokens36/36。
L1/H1/seed3：24/18windows，所有槽真实。
全部window逐source-index审计、split重复性、group/episode overlap=0通过。
坏奇偶anchor split实际共享10episodes；跨界污染拒绝；改变terminal不改变policy输入。
固定T1/L4/H4 loss probe：masked≈1.0000000093，unmasked≈750000.2500000023m²；
padded NaN不影响masked loss，全空mask拒绝。
validation x+100m不改train mean x=0.2545454502105713m；错误all-data mean变47.619049072265625m。
这些只验证dataset与loss边界，不是学习到的模型结果或robot success。
pip check、Markdown本地链接、git diff --check通过；所有输出ignored tmp/。
工程完成时 README双状态已更新；当时 Learning三项未确认。后续学习确认见下文，未开始S0.3。

Run：`.venv/bin/python scripts/s0_2_sequence_dataset.py --output tmp/my_s0_2`，检查report和第一条window。
Modify：预测H4→H2对split/N/history/action shape/有效token数的影响，再新目录运行核对。
Explain：回答学习包五问，特别说明mask不等于非零，以及episode split与reset-group split差别。
Current Task = S0.2；Next Task = S0.3。完成工程后STOP，等待用户学习确认与选择。

## S0.2 learning confirmation — 2026-10-14

用户确认实验和预测均已完成；五项 Explain 准确覆盖因果 history/terminal anchor、
T1/L3/H4 的 mask/index、真实零值与 masked MSE 分母、episode vs reset-group split、
H4→H2 保持22/20anchors及history，target shape变化、valid action tokens48/56→36/36、
train-only真实状态统计避免验证泄漏与重叠/padding偏差。
README和学习包 Run/Modify/Explain 已全部勾选，S0.2 Learning Mastered。
本轮仅同步四份状态文档，git diff --check通过；未重新运行Python实验，runtime沿用上述2026-10-14工程验证。
Current Task = S0.2（已掌握）；Next Task = S0.3。等待用户选择，不自动实现下一任务。

## S0.3 Engineering — 2026-10-14

用户明确请求推进S0.3。启动工作区已有S0.2学习确认的四份文档修改，保留全部工作。
读规则/README/handoff/data contract/S0.2 package；当前只实现本任务。
[学习包](vla/s0_3_rlds_oxe_concepts.md)、[adapter](../data/adapters/rlds_toy.py)、
[experiment](../scripts/s0_3_rlds_concepts.py)、[schema notes](../data/adapters/oxe_schema_notes.json)。

官方source study：RLDS README/rlds_types.py，OXE README，TFDS Fractal/Bridge catalog。
通过git ls-remote HEAD确认并浏览固定revision：
RLDS b35dac3a6b73396b0cb8773095999c4b5d70947c；OXE 9eeb68b989efbcf474e8fb9019e01d02b962a604。
两份catalog version0.1.0，页面更新时间2024-09-03/2024-05-16；2026-10-14查阅。
人工metadata摘要记录language位置/action结构/state/image shape，非下载的真实样本。
未运行producer/builder，未读取TFRecord；schema study不能证明真实episode全符合canonical语义。
官方大notebook未使用（web返回content too large），不需要为本任务下载它。

关键规则：loaded RLDS step action作用于同step observation，最后step action/reward/discount无效。
最后observation的is_terminal映射到内部最后transition；is_last不是success。
本adapter仅synthetic_loaded_rlds_xy/1.0，已知world XY delta[m]/10Hz/ideal point。
要求有效终点、恒定语言、显式success；未知语义/invalid/缺终点不能靠补零修复。
Reward/discount丢弃在metadata中明确记录，本转换不声称全字段无损。

环境重新检查：`.venv/bin/python`，Python3.14.7、NumPy2.5.4，actual NumPy path在本venv内。
未安装新依赖、未下载真实dataset或weights，未修改旧repo。

Executed (final version):

```bash
.venv/bin/python scripts/s0_3_rlds_concepts.py --output tmp/s0_3_verified
.venv/bin/python scripts/s0_3_rlds_concepts.py --ending truncated --output tmp/s0_3_truncated_verified
.venv/bin/python -m pip check
```

两方向ending都PASS：4steps→(4,2)observations/(3,2)actions；末[999,999]与后续NaN占位均被忽略；
terminal/truncated标志正确，success True保持独立，额外False标注也独立保留。
各10/10坏输入拒绝；本地NPZ+JSON round-trip/S0.1 validator/S0.2 window audit通过。
S0.2 L3/H4生成3anchors、6valid action tokens；数值payload94bytes，默认synthetic JSON4039bytes。
pip check、Markdown本地链接、schema-notes JSON读取和git diff --check通过。
以上验证只覆盖toy concept profile，不是TFDS/真实OXE adapter或robot success证据。

Run：默认新目录运行，查看source最后step与local最后transition。
Modify：先预测 `--ending truncated` 对boundary/success/shape/window数量影响，然后运行核对。
Explain：回答学习包五问，特别区分格式统一/语义统一、loaded/original时序与last/terminal/success。
Current Task = S0.3；Next Task = S0.4。Learning三项未勾选，工程后STOP。

## S0.3 learning confirmation — 2026-10-14

用户明确确认实验和预测均已完成；五项 Explain 准确覆盖RLDS/TFDS/OXE职责与格式/动作语义区别，
四观测三转移和末action无效，is_last/is_terminal与terminated/truncated映射及success独立，
原始EnvLogger/dm_env与loaded RLDS时序区别（不再次shift action），
以及Fractal/Bridge真实schema、控制器、frame、单位、终点、success来源与producer约定审计。
README、学习包和roadmap同步为S0.3 Learning Mastered，Run/Modify/Explain全部勾选。
启动pwd正确，git status --short为空；本轮仅修改四份状态文档，未重新执行Python实验、安装依赖或下载数据。
运行环境与工程结果沿用上述S0.3记录（Python3.14.7 / NumPy2.5.4）；不增加真实OXE适配或机器人能力声明。
验证命令：`git diff --check`（通过）；`git diff --stat` / `git status --short`核对四份文档变更。
Current Task = S0.3（已掌握）；Next Task = S0.4。等待用户明确选择，不自动开始下一任务；未commit/push。

## S0.4 Engineering — 2026-10-14

用户明确请求“继续推进S0.4”，仅实现此任务；保留启动时S0.3学习确认的四份文档修改。
pwd正确，读AGENTS/handoff/README/roadmap/data contract/S0.1模板/S0.3 package；S0.4此前没有package。
[学习包](vla/s0_4_real_robot_dataset_inspection.md)、[script](../scripts/s0_4_inspect_robot_sample.py)、
[source manifest](../data/adapters/bridge_sample_manifest.json)。

来源：社区`jesbu1/bridge_v2_lerobot`，revision `3899a5d24770fa593ad09f6f6e22cd417a227431`。
Hugging Face API查schema/单文件大小后，先向用户说明下载大小/用途/必要性，再下载四文件：
meta info4692bytes、task索引1637992bytes、episode0 parquet6384bytes、image_0 video274880bytes，
合计1923948bytes；初次bounded urllib探查和CLI下载验证各一份，总样本传输3847896bytes。
raw在ignored `tmp/s0_4_source`及`tmp/s0_4_download_verified`，四文件SHA256固定在manifest。
只取一episode/一camera，不是原始RLDS/TFDS，不下载全集或weights。
官方Bridge上下文revision经`git ls-remote`核对：`bc60a35b701a12021c8c95e9d8601274d3acd928`。
只读raw-to-numpy和loader，发现日期相关camera latency处理与action relabeling路径；
没有运行producer，也不能证明社区mirror使用哪条路径。mirror许可声明不替代原数据许可审查。

环境修复：启动没有`.venv`。默认Python是Conda base3.14.7；系统`/usr/bin/python3`为3.12.3。
`/usr/bin/python3 -m venv .venv`因缺ensurepip失败，但留下隔离venv解释器。
先说明repair及必要依赖，再用base的pip仅作installer，`--python`明确目标为venv，没有向base/system安装。
Executed environment repair:

```bash
python3 -m pip --python .venv/bin/python3 install --only-binary=:all: 'numpy>=1.26,<3' 'pyarrow>=19,<25' 'av>=14,<18' 'Pillow>=11,<13'
python3 -m pip --python .venv/bin/python3 install pip
.venv/bin/python -m pip check
```

成功Python3.12.3、NumPy2.5.4、PyArrow24.0.0、av17.1.0、Pillow12.3.0、pip26.2.1；
实际解释器`.venv/bin/python3`，sys.prefix为本repo `.venv` / base_prefix `/usr`；
四个实际module路径都位于`.venv/lib/python3.12/site-packages/`，执行前已核对。
wheel新下载约101MB（NumPy16.7/PyArrow48.9/av35.4），Pillow6.9MB使用cache，pip约1.8MB；
先前70MB估算偏低，已向用户更正。requirements仅增加本任务Parquet/CPU AV1/RGB preview依赖。

Executed experiments:

```bash
.venv/bin/python3 scripts/s0_4_inspect_robot_sample.py --output tmp/s0_4_verified
.venv/bin/python3 scripts/s0_4_inspect_robot_sample.py --preview-stride 14 --output tmp/s0_4_stride14_verified
.venv/bin/python scripts/s0_4_inspect_robot_sample.py --download --source tmp/s0_4_download_verified --output tmp/s0_4_download_run
.venv/bin/python scripts/s0_3_rlds_concepts.py --output tmp/s0_4_env_regression
.venv/bin/python -m pip check
git diff --check
```

三次inspection均PASS raw hash、实际Arrow类型/有限值/identity/clock、CPU视频RGB/PTS、8/8损坏输入拒绝。
28行state/action(28,7)float32；timestamp0…5.400000095367432s，最大clock误差1.9073486345888568e−7s。
video CPU decoder libdav1d、5fps、28帧(256,256,3)uint8 RGB；camera mask恒[True,True,True,False]。
task0=`put small spoon from basket to tray`，已查看PNG厨房操作图像；不从视觉判定success。
stride7→14显示5→3帧；独立inline比较source/arrays/video/language/rows/alignment/local_contract均相同。
NPZ训练export未实现：N/N末action语义/有效终点、boundary/success、units/frame/controller/anchor、
producer/original reset group和sensor causality未知，六项blockers明确记录，不能套toy规则。
逐通道action vs state差offset诊断不是时序证明，不执行shift/relabel，也不fit normalization。
新Python3.12环境的S0.3回归（含S0.1 validator/S0.2 windows）通过；pip check无冲突。
Markdown本地链接检查与git diff --check通过；tmp/.venv仍ignored，未改sibling，未commit/push。

Run：新目录运行`--download`（已有source只核验），打开HTML/PNG/report，找last action与blockers。
Modify：先预测preview stride7→14的行号/显示数和raw/clock/shape/diagnostic，再离线运行核对。
Explain：学习包五问，重点是snapshot与producer、N/N与T/T+1、PTS与采集因果、diagnostic与语义。
Current Task = S0.4；Next Task = S0.5。Engineering Complete；Learning三项未勾选，STOP等待用户。

## S0.4 learning confirmation — 2026-10-14

用户明确确认实验和预测均已完成；五项 Explain 准确覆盖固定revision社区转存与原始producer/RLDS证据的区别，
N/N无法确定末action有效性或缺终点（不擅自丢弃/补造boundary），
PTS与记录clock匹配不等于采集/状态/下发同步，不能排除延迟或转存重标时，
逐通道MAE仅提供offset诊断，不凭最小值推断shift/controller，视频外观不替代独立success，
以及stride7→14预览[0,7,14,21,27]→[0,14,27]，raw/5fps/clock/统计/diagnostic/blockers均保持不变。
README和学习包Run/Modify/Explain已全部勾选，roadmap同步S0.4 Learning Mastered。
启动pwd正确、git status --short为空；本轮仅修改四份状态文档，未重新执行Python实验、安装依赖或下载数据。
环境与工程结果沿用上述S0.4记录（Python3.12.3 / NumPy2.5.4 / PyArrow24.0.0 / av17.1.0 / Pillow12.3.0）；
仍未验证原始RLDS/producer语义，未生成内部训练episode，不增加机器人成功或泛化声明。
验证命令：`git diff --check`（通过）；`git diff --stat` / `git status --short`核对四份文档变更。
Current Task = S0.4（已掌握）；Next Task = S0.5。等待用户明确选择，不自动开始下一任务；未commit/push。

## S0.5 Engineering — 2026-10-14

用户明确请求“继续推进S0.5”，只实现当前小任务；保留启动时S0.4学习确认的四份文档修改。
pwd正确，读取AGENTS/README/handoff/roadmap/data contract/S0.1模板与现有S0.1/S0.2代码；S0.5此前无package。
[学习包](vla/s0_5_continuous_action_representation.md)、[script](../scripts/s0_5_action_representation.py)。
独立NumPy脚本，不导入或修改旧仓库，不下载数据/权重，不重解释S0.4 Bridge source。

执行前核对`.venv/bin/python` Python3.12.3，sys.prefix本repo `.venv` / sys.base_prefix `/usr`；
NumPy2.5.4实际module path=`.venv/lib/python3.12/site-packages/numpy/__init__.py`。
requirements只更新覆盖任务注释，无新依赖、安装或环境repair；其它依赖仍仅供S0.4使用。

数据/语义：四独立reset，train servo_0/1/2、validation servo_3；各T4/T+1=5。
q1/q2 rad，links .3/.2m，dt .1s，r_t absolute reference经q_next=q+gain*(r−q)执行。
raw以embodied_episode/1.0存物理单位、normalized=false，S0.1 validator与NPZ+JSON round-trip通过。
measured/previous-reference两种delta、Cartesian absolute/delta五view；位置仅XY FK，无IK或姿态controller。
raw与view分离，各view train-only zscore sidecar保留signature与fit IDs，ddof0，constant std0→scale1原单位，no clip。
S0.2 make_windows/audit_windows复用；normalization只处理真实mask，不用重叠chunks或padding拟合。

Executed final experiments:

```bash
.venv/bin/python scripts/s0_5_action_representation.py --output tmp/s0_5_final_checked
.venv/bin/python scripts/s0_5_action_representation.py --gain 1 --output tmp/s0_5_gain1_checked
.venv/bin/python scripts/s0_5_action_representation.py --horizon 1 --output tmp/s0_5_h1_checked
.venv/bin/python scripts/s0_5_action_representation.py --horizon 6 --output tmp/s0_5_h6_checked
.venv/bin/python -m pip check
git diff --check
```

四次均PASS raw/view/chunk round-trip、correct delta decode与toy servo重放、train-only normalization，11/11拒绝。
default gain.5：q_final≈[.39375,.35625]rad，r_last−q_final≈[.10625,−.15625]rad；四episode toy tracking success均False。
gain1：q_final≈[.5,.2]rad，base最大tracking error1.49e−8rad；四episode toy success均True。
两anchor delta最大差.125→0rad，command vs measured displacement最大差.15625003→0rad。
四步预算末boundary一直truncated；toy success独立；没有learned policy或真实robot success评价。
wrong delta-as-absolute末状态相对正确trace差≈[−.2375,−.43125]rad；全序列复用q0的stale anchor reference最大分量差≈[.1875,.1125]rad。
换link长度same-shaped reference产生XY最大分量差.04878067m；固定local delta误作world最大分量差.20428187m。
metadata错embodiment/order/unit/frame/controller/anchor均拒绝，不将normalized representation统一视为相同语义。

joint_absolute fit12原始训练行，mean≈[.35,.41666668]rad；错误all-data mean≈[.55,.3875]，
validation q1+ .2rad使错误mean变≈[.60,.3875]，train stats不变；validation z最大≈6.86199且未clip。
五view inverse最大float32误差≈7.45e−9原单位；constant训练副本重新servo，std0/scale1/z0通过，原始文件未变。
H3/H1/H6：shape(12,3,2)/(12,1,2)/(12,6,2)，valid tokens27/12/30，anchors均12，normalizers相同。
独立inline手算q trace、train/all-data mean oracle、跨gain/reference/chunk不变与跨H统计比较通过。
pip check无冲突；Markdown本地链接检查、git diff --check通过，全部artifacts/env ignored。
限制：无真实动力学/接触/包角处理，Cartesian view不唯一反推joint，离线delta重放不证明受扰动鲁棒性；
没有model loss/inference latency/泛化或真实robot success声明，S0.4未知语义仍未解决。

Run：新目录默认执行，查看raw、五view、normalizer/chunk与错误解释report。
Modify：先预测gain.5→1对state/delta/reference/joint_absolute stats/chunk/boundary/success的影响，再运行核对。
Explain：回答五问并手算t1 delta两anchor恢复r1。
Current Task = S0.5；Next Task = S0.6。Engineering Complete；Learning三项未勾选，STOP；未commit/push。

## S0.5 learning confirmation — 2026-10-14

用户明确确认实验和预测均已完成；五项 Explain 准确覆盖absolute reference与两种delta anchor的解码，
t1 [.25,.15] / [.2,.2]rad均恢复[.4,.5]rad；command delta与measured displacement/rate区别，
joint/Cartesian依赖运动学、links/frame/unit/姿态/controller且FK(delta)不是Cartesian displacement，
逐表示train-only原始action各一次fit（排除validation/padding/重复chunk）、constant std0/scale1与matching inverse，
shape/z-score不能统一embodiment/order/frame/unit/anchor/controller/dt，以及未来measured-anchor chunk需各自测量。
gain.5→1使本toy测量到reference、两delta一致、四episode success False→True；
absolute references、previous-reference delta、joint_absolute stats、split/chunk/mask和truncated boundary不变。
README和学习包Run/Modify/Explain全部勾选，roadmap同步S0.5 Learning Mastered。
启动pwd正确、git status --short为空；本轮仅修改四份状态文档，未重新执行实验、安装依赖或下载数据。
环境与工程结果沿用上述S0.5记录（Python3.12.3 / NumPy2.5.4）；不增加真实robot success或泛化声明。
验证命令：`git diff --check`（通过）；`git diff --stat` / `git status --short`核对四份文档变更。
Current Task = S0.5（已掌握）；Next Task = S0.6。等待用户明确选择，不自动开始下一任务；未commit/push。

## S0.6 Engineering — 2026-10-14

用户明确请求“继续推进S0.6”，只实现当前任务；保留启动时S0.5学习确认的四份文档修改。
pwd正确，读取AGENTS/README/handoff/roadmap/S0.5 package/code与S0.1 package模板；S0.6此前无学习包。
[学习包](vla/s0_6_discrete_action_tokenization.md)、[script](../scripts/s0_6_action_tokenization.py)。
执行前确认`.venv/bin/python` Python3.12.3、sys.prefix为repo `.venv` / base_prefix `/usr`，
NumPy2.5.4实际module path=`.venv/lib/python3.12/site-packages/numpy/__init__.py`。
无安装/新依赖/下载；requirements仅更新任务注释；未读改sibling或S0.4原始Bridge数据。

复用S0.5 explicit generator与round_trip/servo/views，raw为四独立reset，train3/validation1，各4commands/5observations。
joint_absolute rad、dt.1s、raw gain.5不变；range只fit12原始train actions，low≈[.1,.2] / high≈[.6,.65]rad。
scalar uniform B-bin，每channel namespace j*B+bin；内部edge归右，low/high范围内，decode中心float32。
overflow clip返回per-scalar bool saturation，reject模式测试；constant range拒绝，不假造范围。
IDs/serialized/decoded/saturation单独保存，time-major每episode8IDs；int64 ID payload64bytes vs raw action32bytes，非压缩。
codec保存signature/fit IDs/规则/B/ranges，NPZ allow_pickle=False往返与saved codec decode通过。

Executed final experiments:

```bash
.venv/bin/python scripts/s0_6_action_tokenization.py --output tmp/s0_6_final
.venv/bin/python scripts/s0_6_action_tokenization.py --bins 16 --output tmp/s0_6_b16_final
.venv/bin/python -m pip check
git diff --check
```

B8/B16均PASS raw/token/codec round-trip、ID decode→encode稳定、manual B4 oracle、
endpoint/内部edge/nextafter/out-of-range boundary oracle、每通道半格误差界与16/16拒绝。
train saturation0/24；validation saturation5/8(62.5%)，bins翻倍不改变range或saturation。
half-bin界[.03125,.028125]→[.015625,.0140625]rad；base MAE≈[.0125,.01875002]→[.00624999,.00703125]rad。
held-out最大total error≈[.73124993,.128125]→[.71562493,.11406250]rad，
clipping max≈[.69999993,.10000000]rad不变；half-bin只适用于原始in-range值的total error。
错误all-data high≈[1.3,.65]，validation q1+.2rad后变≈[1.5,.65]，正常train high保持≈[.6,.65]。

Counterfactual toy replay以original r_last为目标，阈值max tracking error≤.02rad，四命令budget仍truncated。
gain1 train baseline3/3，两B下decoded0/3→3/3；validation baseline1/1而decoded始终0/1。
gain.5 train/validation合计baseline0/4、decoded0/4；失败stage为final joint tracking threshold，四条全保留。
这里不是训练预测模型、真实robot success、OOD泛化或大型tokenizer复现；没有model loss/部署latency声明。
增大B只减小in-range最坏界、词表16→32，序列仍每episode8IDs，任意单点误差不保证单调。

独立inline检查raw与codec fit/ranges/ID序列/半格界、B变化不变量、全分母toy scores；
B2/B3/B65536额外boundary/ID stability小probe通过；pip check、本地Markdown links与git diff --check通过。
artifacts/.venv保持ignored；raw契约未升级、S0.5源码未改，未commit/push。
限制：仅two-channel scalar codec，无constant-channel扩展、wrapped angles、gripper、padding/special tokens、learned prediction；
已知toy反馈重放不证明真实控制、sensor causality、扰动鲁棒性或泛化，S0.4未知语义仍未解决。

Run：新目录默认运行，查看raw/codec/IDs/decoded/saturation/replay report。
Modify：先预测B8→16范围/词表/序列长度/误差界/saturation/toy results，再运行核对。
Explain：学习包五问，手算B4 oracle并说明half-bin与range外误差区别。
Current Task = S0.6；Next Task = S0.7。Engineering Complete；Learning三项未勾选，STOP等待用户。

## S0.6 learning confirmation — 2026-10-14

用户明确确认实验和预测均已完成；五项 Explain 准确覆盖train-only每通道[L,H]与B等宽bins，
channel ID=jB+b、中心decode、内部edge归右与端点有效，以及clip+saturation/reject区别；
physical往返有损、合法ID在同codec下往返稳定，half-bin界仅约束in-range原值或clipped值，
range外总误差需包含clipping；fit只用12原始train动作，排除validation/padding/重叠chunk。
B8→16词表16→32、每episode仍8tokens、half-bin界减半，actual MAE下降不是普遍单调定理，
validation仍5/8饱和；ID0合法且不是PAD，错误channel namespace拒绝。
用户正确区分original-reference评分与decoded-target评分，准确报告gain1 train baseline3/3、decoded0/3→3/3，
validation baseline1/1而decoded均0/1、gain.5合计0/4；toy不证明真实机器人/VLA策略成功。
README和学习包Run/Modify/Explain全部勾选，roadmap同步S0.6 Learning Mastered。
启动pwd正确、git status --short为空；本轮仅修改四份状态文档，未重新执行实验、安装依赖或下载数据。
环境与工程结果沿用上述S0.6记录（Python3.12.3 / NumPy2.5.4）；不增加真实robot success或泛化声明。
验证命令：`git diff --check`（通过）；`git diff --stat` / `git status --short`核对四份文档变更。
Current Task = S0.6（已掌握）；Next Task = S0.7。等待用户明确选择，不自动开始下一任务；未commit/push。

## S0.7 Engineering — 2026-10-14

用户明确请求“推进S0.7学习”，按既有Sprint流程完成此工程/学习包；仅当前任务，不启动Stage1。
pwd正确；保留启动时S0.6学习确认的四份文档修改。读取规则/README/handoff/roadmap/data contract/S0.1模板与S0.2/S0.5源码。
[学习包](vla/s0_7_action_chunk_representation.md)、[script](../scripts/s0_7_action_chunk.py)。
环境执行前确认`.venv/bin/python` Python3.12.3、sys.prefix repo `.venv` / base_prefix `/usr`，
NumPy2.5.4与预览前额外核对Pillow12.3.0，实际module均在`.venv/lib/python3.12/site-packages/`。
没有新依赖、安装、数据/weights下载，requirements只更新task注释；NumPy计算、已有Pillow生成PNG。
未读改旧repo或真实Bridge source，S0.1/S0.2/S0.5源码保持不变。

数据：3独立scene resets，各红/蓝literal指令一条；group0 T2/group1 T6为train4episodes16actions，
group2 T4为validation2episodes8actions。world XY point displacement[m]/dt.1s，每component最大.06m。
raw T+1 state/image/clock，budget均truncated，独立success3/6，5行合法zero-vector action，失败数据保留。
32×32 uint8 RGB，u=16+x/.02,v=16−y/.02，静态red[.24,0] / blue[−.24,0]m，非真实camera。
goal覆盖重叠agent marker，位置由state提供；image目标由精确颜色与calibration解析，不是learned vision。
S0.2窗口复用；S0.7 context增加anchor image/image_index并审核，未放宽旧state-only loader。
rule只接causal history/mask、current RGB/language、H，不接action mask/future/results；history仅用最新state。
known ideal-point unroll生成H delta proposals，无embedding/fusion/neural head或learned world model。

Executed final experiments:

```bash
.venv/bin/python scripts/s0_7_action_chunk.py --output tmp/s0_7_checked
.venv/bin/python scripts/s0_7_action_chunk.py --execute 4 --output tmp/s0_7_e4_checked
.venv/bin/python scripts/s0_7_action_chunk.py --horizon 2 --output tmp/s0_7_h2_checked
.venv/bin/python scripts/s0_7_action_chunk.py --history 1 --horizon 1 --output tmp/s0_7_h1_checked
.venv/bin/python -m pip check
git diff --check
```

四次均PASS raw/chunk NPZ+JSON round-trip/S0.1 validator/S0.2 indices、RGB/state相同不同literal指令的rule chunks，
future/result anchor0 intervention、NaN padding不影响masked loss、12/12拒绝检查。
default L2/H4：train16/validation8anchors；state(16,2,2)/(8,2,2)、RGB(16,32,32,3)/(8,32,32,3)，
target(16,4,2)/(8,4,2)，valid tokens42/20、masked MSE0/0m²，scalar denominators84/40。
H2 valid tokens28/14，L1/H1为16/8；raw/split/context/anchor不因H/E变化改动，L1无history padding。
末index train[5,-1,-1,-1]/validation[3,-1,-1,-1]，mask[T,F,F,F]，真实zero与padding区分。
独立inline核对T2 anchor1 history[0,1]/action[1,-1,-1,-1]、source RGB、raw5行zero及手算rollout trace。

四步rollout固定起点[0,-.12]、红/蓝两条，H4，第二次transition后同样+.12m world x扰动；budget始终truncated。
E1/E4 nominal均2/2；disturbed为1/2与0/2；每条rule calls4→1。
red受扰后恰好到goal，E1新观测停住、E4旧suffix overshoot .12m；blue被推回，剩余预算不足，E1也失败 .12m。
失败stage均final goal error after command budget，所有commands/trace/decision ticks保留；不隐去失败。
single-run rule plan latency各rollout记录median/max/sample count，排除render/controller，非neural inference或部署benchmark。
E只改actual prefix，不改offline H/mask/loss；offline0不是训练模型结果或扰动成功保证。
已查看context_preview.png，确认三组同场景paired指令与agent/target显示正确；caption不进入policy image。

独立跨E/H raw/context/shape/mask/count/trace不变量检查、local Markdown links、pip check、git diff --check通过。
artifacts/env仍ignored，未commit/push；Stage0 Engineering全部完成，但S0.7 Learning仍未确认。
限制：精确marker与literal语言、ideal point、known unroll，不涉及camera delay/real dynamics/contact/learned language或vision。
toy两条件完整分母不是真实robot benchmark；短E也会因budget失败，不声称一般鲁棒性/泛化；未自动开始Stage1。

Run：默认新目录，打开PNG/report/NPZ，检查masks与nominal/disturbed traces。
Modify：先预测H4固定E1→4对dataset/feedback/calls/success的影响，再运行核对。
Explain：学习包五问，手算T2 anchor1并说明red/blue不同失败机制。
Current Task = S0.7；Next Task = S1.1。Engineering Complete；Learning三项未勾选，STOP等待用户。

## S0.7 learning confirmation — 2026-10-14

用户明确确认实验和预测均已完成；五项 Explain 准确覆盖causal state history float32 world m、
anchor RGB uint8和episode language条件，planner不读取future state/image/action、success或末端label mask；
T2/L2/H4 anchor1 history[0,1]/image index1/action[1,-1,-1,-1]及相应mask，真实zero有效性按source index判断；
H/E/dt的输出/执行/时间含义，H4下E1→4不改dataset/labels/mask/offline loss、calls4→1与反馈.1→.4s。
用户正确解释offline MSE0不代表闭环成功，red E1重新观察停住、E4 stale suffix overshoot，
blue因扰动与剩余预算不足连E1也失败，stage为budget后的final goal error；disturbed成功1/2→0/2。
精确marker、literal language和known point dynamics是手写规则，paired条件响应/因果隔离/toy rollout
不证明真实视觉语言理解、模型泛化或部署鲁棒性。
README与学习包Run/Modify/Explain全部勾选，roadmap同步S0.7 Learning Mastered，Stage0全部工程与学习完成。
启动pwd正确、git status --short为空；本轮仅修改四份状态文档，未重新执行实验、安装依赖或下载数据。
环境与工程结果沿用上述S0.7记录（Python3.12.3 / NumPy2.5.4 / Pillow12.3.0）；不增加learned VLA或真实robot能力声明。
验证命令：`git diff --check`（通过）；`git diff --stat` / `git status --short`核对四份文档变更。
Current Task = S0.7（已掌握）；Next Task = S1.1。等待用户明确选择，不自动开始Stage1；未commit/push。

## S1.1 Engineering — 2026-10-17

用户授权“继续推进S1.1的学习”，仅推进本任务。启动pwd正确，git status --short为空。
读取AGENTS/README/roadmap/handoff、S0.1模板与S0.7包及相关源码；此前无S1.1包。
新增[学习包](vla/s1_1_language_conditioned_state_policy.md)与[script](../scripts/s1_1_language_state_policy.py)。
CPU NumPy 4→32 tanh→2 MLP，226参数，手写backprop/Adam；固定one-hot，不实现S1.2。
30 reset groups、左右paired共60episodes，每条T10/T+1；train24groups/480actions，validation6groups/120actions。
固定scale .3m/.06m，不fit统计；raw保留physical world delta、10Hz、末truncated与独立success。
无语言同结构重新训练，输入language位清零；固定epochs不做validation调参。

执行前核实repo .venv Python3.14.7、NumPy2.5.4，sys.prefix和numpy actual path都在repo .venv。
与此前3.12记录不同，以本轮实测为准；未安装/下载，未来Torch兼容未验证。
未改旧repo，无commit/push；生成数据、权重、日志均ignored tmp。

执行命令（最终代码另以final目录重跑）：

```bash
.venv/bin/python scripts/s1_1_language_state_policy.py --output tmp/s1_1_verified
.venv/bin/python scripts/s1_1_language_state_policy.py --epochs 20 --output tmp/s1_1_short_verified
.venv/bin/python scripts/s1_1_language_state_policy.py --output tmp/s1_1_final
.venv/bin/python scripts/s1_1_language_state_policy.py --epochs 20 --output tmp/s1_1_short_final
.venv/bin/python -m pip check
git diff --check
```

默认conditioned train/validation raw MSE=1.04754e−6/1.34870e−6m²，toy success12/12；
no-language=4.28348e−4/4.66927e−4m²，6/12。20步分别3/12、1/12。
同state[0,0]含语言x≈−.06/+.06m，无语言两次+.003695m。
所有失败均保留trace/commands/error/stage（10步预算终点误差）；report另存clip后MSE与120次推理median/p95。
有限差分逐参数gradient最大差1.40e−11；raw contract/transition/final expert goal、round-trip、
reset overlap=0、model reload通过；最终脚本另拒绝unknown literal/float64/nonfinite三个输入。
限制：同分布6新reset，固定两目标/literal/ideal dynamics，不是paraphrase/vision/real robot泛化证据。
Learning未勾选；下一步用户Run→预测20epochs→Modify→回答学习包五问。Next S1.2只计划，STOP。

## S1.1 learning confirmation — 2026-10-17

用户明确确认实验和预测均已完成，五项 Explain 准确覆盖state/one-hot输入，
physical与dimensionless动作尺度、T/T+1及H=E=1；literal条件响应不等于自然语言泛化；
reset-group split与固定尺度/train-only fitted normalizer区别；无语言策略的位置相关性与MSE折中。
用户给出确定性无语言基线的6/12上界：同reset两指令产生相同轨迹，无法同时达到两个相距.48m的目标。
正确说明2000→20只改变更新次数，以及offline loss、闭环分布偏移、失败stage和延迟的评价区别。
补充边界：报告的encode+MLP+clip计时不能单独证明完整系统满足.1s控制周期。
README和学习包Run/Modify/Explain全部勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮只改四份状态文档，未执行Python实验、安装或下载。
实验环境/结果沿用上节Python3.14.7 / NumPy2.5.4记录，不增加真实robot或泛化声明。
验证：`git diff --check`通过；`git diff --stat`及`git status --short`核对四份文档。
Current Task = S1.1（已掌握）；Next Task = S1.2（仅计划），等待用户选择；未commit/push。

## S1.2 Engineering — 2026-10-17

用户明确授权“继续推进S1.2”。启动pwd正确，已有S1.1学习确认的四份文档修改，全部保留。
读取AGENTS/README/handoff/roadmap、S1.1包/代码和S0.1模板；S1.2原无学习包。
新增[学习包](vla/s1_2_tiny_language_representation.md)、[script](../scripts/s1_2_language_representation.py)。
只当前任务，不进入S1.3/S1.6；没有浏览外部资料、数据/权重下载或依赖安装。
环境执行前确认repo .venv Python3.14.7，NumPy2.5.4，actual path在.venv/lib/python3.14/site-packages/numpy/__init__.py。
本轮NumPy可运行，未来Torch兼容仍需另核验；requirements只改任务注释。

复用S1.1 raw generator（旧脚本未改），60episodes、30paired reset groups，train480/validation120actions。
read_batches从raw NPZ+JSON读取，不用generator附带的one-hot derived features。
train-only lower/split词表PAD0/UNK1/left2/move3/right4；max4tokens，空/过长拒绝，不截断。
4维embedding masked mean + p/.3→6→32 tanh→2，310参数，手写embedding反向/Adam，输出×.06m再clip。
初始PAD/UNK零；PAD无梯度，UNK无训练样本；重复ID用np.add.at累加。
masked mean丢失词序，零UNK仍进入分母；map/reject只改推理，不改训练词表/权重。

Executed:

```bash
.venv/bin/python scripts/s1_2_language_representation.py --output tmp/s1_2_verified
.venv/bin/python scripts/s1_2_language_representation.py --unk-policy map --output tmp/s1_2_map_verified
.venv/bin/python scripts/s1_2_language_representation.py --output tmp/s1_2_final
.venv/bin/python scripts/s1_2_language_representation.py --unk-policy map --output tmp/s1_2_map_final
.venv/bin/python -m pip check
git diff --check
```

两模式train/validation raw MSE=1.374777e−6/1.606606e−6m²，validation executed MSE=1.291406e−6m²；
known held-out6resets共12条success12/12。四类OOV每类6条共24，reject全部拒绝、success0/24；
map无拒绝、success0/24，全部10步终点误差，保留全部trace/commands/error/stage。
probe goal是评价器人工约定，不传policy；port/starboard共同ID[3,1,0,0]，同state动作相同。
please+原句在零UNK下句向量缩放2/3；right!因保留标点也变UNK。
known局部推理120samples，map OOV240samples，report保存median/p95；reject OOV无推理计时null。
梯度finite difference最大差9.01e−13，含重复词；PAD NaN不变、词序置换、零UNK缩放、碰撞和7拒绝检查通过。
raw契约/转移/终点/round-trip复用S1.1；tokenizer/weights/token arrays重载逐值一致。
独立inline验证跨UNK模式metrics/history/weights/tokens一致、词表仅train、map轨迹加法和动作限幅通过。
Markdown本地链接检查、pip check和git diff --check通过；输出/.venv ignored，未commit/push。
限制：一个seed、两种训练文本、固定目标/ideal point；不是同义词、语法、真实robot或VLA泛化证据。
Current Task S1.2工程完成，Learning未勾选；Next Task S1.3仅计划。
用户下一步Run→先预测reject/map不变量和OOV机制→Modify→回答五问，STOP等待用户。

## S1.2 learning confirmation — 2026-10-17

用户明确确认实验和预测均已完成；五项 Explain 准确覆盖tokenization/embedding/pooling/state融合/
物理action的数据流、shape/dtype/unit、310参数和ID不可微；PAD与UNK的mask区别及零UNK平均分母。
用户正确手算please move right的句向量为原句2/3；指出port/starboard碰撞后同state输入及确定性轨迹相同，
增加MLP更新不能恢复编码时已丢失的信息；说明词序不变性与重复词的相对权重变化。
准确区分reject/map保持训练/known结果不变，0/24分别为tokenization拒绝和10步执行后的终点失败；
明确输入接受、动作产生与语义理解三个层次，未过称自然语言泛化。
README与学习包Run/Modify/Explain全部勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅改四份状态文档，未重跑实验、安装或下载。
环境与工程结果沿用上述S1.2记录（Python3.14.7 / NumPy2.5.4），能力限制保持不变。
验证命令：`git diff --check`（通过）、`git diff --stat`、`git status --short`（仅四份文档）。
Current Task = S1.2（已掌握）；Next Task = S1.3（仅计划），等待用户选择；未commit/push。

## S1.3 Engineering — 2026-10-17

用户授权“继续推进S1.3”。启动pwd正确，已有S1.2学习确认四份文档修改，全部保留。
读取AGENTS/README/handoff/roadmap、S1.2包和S0.1 validator，当前无S1.3包。
新增[学习包](vla/s1_3_tiny_visual_encoder.md)、[script](../scripts/s1_3_visual_encoder.py)，仅S1.3，不接语言/fusion。
环境确认repo .venv Python3.14.7（base_prefix miniconda3但sys.prefix本repo隔离venv）、NumPy2.5.4，
actual NumPy path在.venv/lib/python3.14/site-packages/numpy/__init__.py。
初始检查import PIL发现ModuleNotFoundError；无需Pillow，向用户解释后采用stdlib PNG，无修复/安装/下载。
requirements只改覆盖任务注释，不移除旧实验的Pillow依赖；当前pip check仅证明已装依赖无冲突，不证明全部历史任务可跑。

数据为synthetic_visual_scene/1.0，独立static场景，不是episode训练集；输出dataset_contract.json显式记录。
120个unique red中心，train96/validation24，scene/target中心不重叠，blue位置可跨split复用。
RGB32×32 uint8，red/blue3×3 squares，固定背景/颜色、odd center3…27、中心距离>=6pixels。
CNN学27×4 kernel+4bias+4head共116参数，3×3 stride2 valid→15×15×4 tanh→空间heatmap。
中心格CE监督，full batch400epochs Adam lr.03 seed7；soft-argmax grid和world映射固定，非learned action policy。
feature map/tokens/GAP输出，label仅loss/评价；CNN只读当时RGB。

Executed:

```bash
.venv/bin/python scripts/s1_3_visual_encoder.py --output tmp/s1_3_verified
.venv/bin/python scripts/s1_3_visual_encoder.py --observation occluded --output tmp/s1_3_occluded_verified
.venv/bin/python scripts/s1_3_visual_encoder.py --output tmp/s1_3_final
.venv/bin/python scripts/s1_3_visual_encoder.py --observation occluded --output tmp/s1_3_occluded_final
.venv/bin/python -m pip check
git diff --check
```

visible validation world MSE5.18778e−8m²，mean norm error.000309335m/.0154668pixels，peak24/24；
train MSE4.90538e−8m²。train-mean baseline validation MSE.025387m²。
每条件24条10步rollout、dt.1s、world delta clip.06m、same measured start[0,-.12]，final max error<=.01m。
visible24/24，occluded0/24，blank0/24，oracle24/24；全部commands/trace/estimates/error/stage保存，末budget truncated。
scene静态、agent不画在RGB，state每次真实读取；不是moving-camera/动态视觉feedback测试。
每条件240次CNN+controller局部推理保存median/P95，不含render/environment；oracle跳过CNN。
flag只选主报告条件，三对照每次全部保存，原始目标不因遮挡更改。
GAP target内部平移最大差5.96e−8，而predicted x跟随+.12m；R/B交换目标转到原blue处，max定位误差.000234m。
finite difference全部116参数最大差2.19e−11；patch stride/order、pixel-world oracle与5拒绝输入通过。
raw/weights/features round-trip通过；独立inline核对跨flag原始数组/weights/features相同、rollout加法/T+1/限幅/全分母。
已查看preview.png：六行RGB/相对归一化heatmap/遮红，目标与热图一致；预览不是置信度图。
Markdown本地链接、pip check和git diff --check通过；artifacts/.venv ignored，无commit/push或旧repo修改。
限制：一个seed、精确颜色/尺寸/格点与已知相机、ideal dynamics；不是real robot/VLA泛化或亚像素真实精度证据。
当前只Engineering Complete；用户下一步Run→预测visible/occluded不变量→Modify→五问Explain，Learning不勾选。
Current Task S1.3；Next Task S1.4仅计划，STOP等待用户。

## S1.3 learning confirmation — 2026-10-17

用户明确确认实验和预测均已完成；五项 Explain 准确覆盖RGB→patch→CNN→空间tokens→heatmap→
pixel uv→world goal→delta controller的数据流、shape/dtype/unit/frame与15×15计算。
准确列出116可学习参数，区分固定soft-argmax/grid/calibration/controller，以及CE与物理world MSE。
正确解释GAP在指定平移probe中丢位置、225 tokens为空间而非时间，索引j→r/c→pixel center。
准确区分可见/遮红推理条件变化与训练/split/weights/真实目标不变量，softmax输出不是目标存在检测。
补充记录：CLI两次均执行三条件，flag仅选主报告；visible特征与各既定条件结果不因flag改变。
用户正确界定24/24、.31mm为固定renderer证据，解释oracle可达、blank无信息仍输出、split与latency的实际限制。
README与学习包Run/Modify/Explain全部勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅改四份状态文档，未重跑实验、安装或下载。
环境/工程证据沿用上述S1.3记录（Python3.14.7 / NumPy2.5.4），不增加真实robot或视觉泛化声明。
验证命令：`git diff --check`（通过）；`git diff --stat`、`git status --short`（仅四份文档）。
Current Task = S1.3（已掌握）；Next Task = S1.4（仅计划），等待用户选择；未commit/push。

## S1.4 Engineering — 2026-10-17

用户授权“继续推进S1.4”。启动pwd正确，保留已有S1.3学习确认四文档修改；读AGENTS/README/
handoff/roadmap/S1.3包与代码，本任务原无package。不启动S1.5，不修改旧源码或sibling repo。
新增[学习包](vla/s1_4_multimodal_fusion.md)、[script](../scripts/s1_4_multimodal_fusion.py)。
执行前确认repo .venv Python3.14.7、NumPy2.5.4，实际numpy path为.venv/lib/python3.14/site-packages/numpy/__init__.py。
只用现有NumPy/stdlib；无依赖安装、数据/weights下载；requirements只改task注释。

复用S1.3确定scene generator，train96/validation24场景，scene及red中心不重叠，blue中心可重复。
每scene4个随机当前XY（seed23）×2literal color条件，train768/validation192独立snapshot；不是训练episode。
先只用train red监督重训116参数CNN400步，随后冻结；同一CNN在R/B通道交换RGB上检测blue，显式颜色先验。
视觉输入是两个预测world XY summary，不是真实label或完整225patch token；language为两literal one-hot，不宣称文本语义学习。
raw4tokens×4features=[red,blue,language,state]，fixed XY/.3m、色marker；两模型共享表示。
concat flatten16→32 tanh→2共610参数；Transformer4→8+type4×8，pre-LN单headQ/K/V8，
attention4×4+残差，FFN8→16 tanh→8+残差，state token→2共618参数；LN无affine。
训练proposal=真实selected goal−current state，m，输出×.3m恢复；执行才clip±.06m/dt.1s。
独立fusion_contract明确proposal不是executed action，不改source scene_contract或raw物理数据。
full batch1600epochs Adam lr.003/seed7，验证集不调checkpoint；无框架/autograd。

Executed:

```bash
.venv/bin/python scripts/s1_4_multimodal_fusion.py --output tmp/s1_4_verified
.venv/bin/python scripts/s1_4_multimodal_fusion.py --kick-x .12 --output tmp/s1_4_kick_verified
.venv/bin/python scripts/s1_4_multimodal_fusion.py --output tmp/s1_4_final
.venv/bin/python scripts/s1_4_multimodal_fusion.py --kick-x .12 --output tmp/s1_4_kick_final
.venv/bin/python -m pip check
git diff --check
```

前两次初版attention因np.sqrt(8)的np.float64 scalar把中间运算提升到float64，初步Transformer23/48；
检查dtype后修成Python常数8**.5，加入runtime float32断言，最终两次重跑结果才作为正式证据。
最终concat train/validation proposal MSE=1.85651e−5/3.65902e−5m²，validation executed MSE4.85314e−6m²；
Transformer=2.34639e−4/2.46857e−4m²，executed4.69856e−5m²。数值精度敏感，不能称多seed稳定结论。
每条件24scene×2color=48条，10commands/11states，起点[0,-.12]、dt.1s、末maxerror<=.01m。
concat full44/48、Transformer8/48；no_vision/no_language/no_state各0/48，保留全部trace/proposals/commands/error/stage。
所有失败stage=final goal error after 10-step budget，末均truncated，success独立；concat full error median/max.004079/.017667m，
Transformer .016509/.053685m。oracle与冻结视觉+手写颜色路由48/48，nominal/kick均如此。
+.12m外扰发生第4次转移后，只改physical state，下一步真实feedback；最终两模型full成功数仍44/48与8/48。
三模态消融为推理后编码内容置零、保留type/markers，未重训no-modality baseline；0/48不能单独证明语义理解。
每模型每条件480次推理记录median/P95，含两CNN+tokens+fusion+clip，不含render/environment；消融仍跑CNN。
参考controller用cached视觉，不做latency比较；静态RGB无agent，未测试动态视觉反馈。

全部fusion参数finite difference最大差concat1.03e−11、Transformer2.02e−9，5拒绝输入通过；
raw/fusion arrays和model reload通过，CNN冻结逐值验证。独立inline检查跨kick权重/data/attention/offline不变、
no_language pair输入/输出相同、attention row sum1、runtime float32、rollout外扰时序/T+1/加法/限幅/完整分母通过。
已查看scenes.png，确认红蓝目标场景；本地Markdown links、pip check、git diff --check通过。
artifacts/.venv ignored，无commit/push。限制：一个seed、两literal、固定renderer/颜色先验、summary token，
不是raw-image/text端到端VLA、language泛化或真实robot证据；更复杂架构没有在本设置中更好，失败如实保留。
Current Task S1.4 Engineering Complete，Learning未勾选；Next Task S1.5仅计划。
下一步用户Run→先预测kick不变量/实际反馈→Modify→回答包中五问，STOP等待用户。

## S1.4 learning confirmation — 2026-10-17

用户明确确认实验和预测均已完成；五项 Explain 准确覆盖CNN红蓝预测/通道交换先验、真实当前state、
literal条件与四角色token，监督标签隔离及left/right embedding无法直接声称color语义迁移。
准确列出concat610/Transformer618参数及逐层shape，解释type embedding、Q/K/V、key轴softmax、
pre-LN、residual、FFN与state readout；区分proposal与executed delta，loss无量纲、MSE单位m²和.09换算。
正确解释三模态置零为推理干预、不重训，no-language配对同输入同轨迹不能同时达到分离目标，
0/48仅为指定干预的证据，不等于一般语义理解或无模态重训失败。
明确kick在tick3执行后/第4次转移后加入world X，下一步才读取真实受扰state；训练与offline不变。
正确界定44/48与8/48是float32单seed配置结果、Transformer训练误差亦高，oracle/视觉参考48/48与局部latency的限制。
README与学习包Run/Modify/Explain全部勾选，roadmap同步Learning Mastered。
启动pwd正确、git status --short为空；本轮只改四份状态文档，未重跑实验、安装或下载。
环境/实验沿用上节S1.4记录（Python3.14.7 / NumPy2.5.4）；不增加VLA泛化或真实robot能力声明。
验证命令：`git diff --check`（通过）；`git diff --stat`、`git status --short`（仅四份文档）。
Current Task = S1.4（已掌握）；Next Task = S1.5（仅计划），等待用户选择；未commit/push。

## S1.5 Engineering — 2026-10-18

用户授权“继续推进S1.5的学习”。启动pwd正确、git status --short为空；读AGENTS/README/handoff/
roadmap/S1.4学习包和代码、S0.2窗口源码，无现存S1.5包。仅本任务，不进入S1.6/S1.7。
新增[学习包](vla/s1_5_tiny_transformer_vla.md)、[script](../scripts/s1_5_tiny_transformer_vla.py)。
执行前确认repo .venv Python3.14.7、NumPy2.5.4，actual path在.venv/lib/python3.14/site-packages/numpy/__init__.py。
无安装/数据或weights下载；stdlib PNG，requirements只改任务注释；不改旧脚本或sibling repo。

新patch-center renderer：RGB32×32，red/blue3×3 squares，中心{4,12,20,28}²，fixed .02m/pixel。
240有序不同中心layouts中seed11抽40，train32/validation8布局不重叠，单物体中心允许复用。
每layout红蓝2episode，T8/9observations，dt.1s、world delta clip.06m、预算truncated、success独立。
raw embodied_episode/1.0 round-trip后，复用S0.2 history1/H3 causal windows/audit；train64episodes512anchors，
validation16episodes128anchors，valid action tokens1344/336。raw physical units不覆盖，label mask不入policy。
16个8×8meanRGB/255+fixed xy网格patch features→learned projection；3个train-only word embeddings；1个state投影。
context20×16，单pre-LN attention+FFN，颜色词index18 readout，parallel head H3×2 actual delta，输出×.06m后clip。
所有2454参数由masked action MSE联合训练，没有CNN定位/颜色swap/object truth/camera goal-error controller进入policy。
Adam lr.002、batch32、2400updates，init7/sampler8；一seed，不按validation挑checkpoint，但原型曾共用validation。

Executed:

```bash
.venv/bin/python scripts/s1_5_tiny_transformer_vla.py --output tmp/s1_5_verified
.venv/bin/python scripts/s1_5_tiny_transformer_vla.py --output tmp/s1_5_color_readout
.venv/bin/python scripts/s1_5_tiny_transformer_vla.py --output tmp/s1_5_final
.venv/bin/python scripts/s1_5_tiny_transformer_vla.py --execute 3 --output tmp/s1_5_e3_final
.venv/bin/python -m pip check
git diff --check
```

首版proprio readout train MSE5.010e−4、closed-loop0/16；发现单block query尚未带语言条件，
改成颜色词readout、同数据/预算重跑，仍0/16；不声称结构修改解决了问题或更优。
最终2400steps train raw/clipped MSE5.49110e−4/5.47313e−4m²，validation5.66041e−4/5.64488e−4m²；
zero baseline train9.83482e−4/validation6.01190e−4m²，验证只略优于零动作。不同指令chunk数值不同不足证明grounding。
E1与E3 nominal/disturbed均0/16，oracle均16/16；第2次transition后world x+.12m。
E1 calls8，E3 calls3（prefix3/3/2），feedback.1→.3s，8steps/9states，末maxgoalerror<=.01m。
nominal median/max terminal error E1 .24250/.40148m、E3 .25048/.39201m；disturbed .31005/.54857与.31625/.47499m。
完整failure stage=final goal error after 8-step budget，所有trace/commands/decision chunk、goal/error/success/truncated保留。
report逐rollout记录局部latency median/P95，含patch/text/model/clip，不含render/environment；oracle跳过模型。
最终模型欠拟合/闭环失败，不将Engineering Complete表述成task success或Learning Mastered。

最终全2454参数finite difference最大误差1.08e−10；padding NaN不改loss/grad、5拒绝输入通过。
raw/schema/physical transition/round-trip、window source-index/scene-group split、weights reload通过。
独立inline确认跨E数据/weights/curve/offline/attention逐值不变、attention float32/row sum1、
T/T+1/actual prefix/budget/clip/外扰时序/加法/完整分母通过；train有效chunk中768行真实zero，末mask[T,F,F]/idx[7,-1,-1]。
已查看scenes.png，四个validation红蓝格点layout正确；Markdown本地链接、pip check、git diff --check通过。
artifacts/.venv保持ignored，未commit/push。限制：mean-patch丢内部位置、只有两literal、单seed、静态RGB/ideal point，
不是预训练VLA/真实robot/语言泛化证据；oracle成功与梯度PASS不能补救模型失败。
Current Task S1.5 Engineering Complete（失败实验），Learning三项未勾选；Next S1.6仅计划。
下一步用户Run→预测E1/E3不变量与feedback→Modify→五问Explain，STOP等待用户；不自动扩张修复或进入后续任务。

## S1.5 learning confirmation — 2026-10-18

用户明确确认实验和预测均已完成，五项 Explain 准确覆盖RGB平均patch/固定坐标、train-only word embedding、
proprio投影、20×16 context和2454参数，区分固定预处理与learned action-loss通路。
正确解释index18颜色词query的语言条件与单block proprio query限制；明确修改后仍0/16、train loss更高，不能称有效。
准确手算末anchor7 mask/index、单样本分母2、train1344/2688及validation336/672有效token/scalar分母，
区分实际delta chunk与S1.4 proposal，并保留真实zero labels。
正确说明E1→3不改data/tokenizer/weights/H/mask/offline，calls8→3、prefix3/3/2、反馈最多.1→.3s，
第2次转移后外扰，E1在tick2重观测、E3到tick3；同0/16不表示trace/error相同。
用户用train/validation loss、zero baseline与oracle16/16准确界定当前模型未学会控制；
欠拟合/表示/优化/漂移需要受控排查，梯度PASS、不同指令输出和局部latency均不能证明能力达标。
README/学习包Run/Modify/Explain勾选，roadmap同步Learning Mastered；0/16闭环失败结论不变。
启动pwd正确，git status --short为空；本轮仅改四份文档，未重跑实验、安装或下载。
环境/工程证据沿用上节（Python3.14.7 / NumPy2.5.4），无新增成功/泛化声明。
验证命令：`git diff --check`（通过）；`git diff --stat`、`git status --short`（仅四份文档）。
Current Task = S1.5（学习已掌握，模型控制仍失败）；Next Task = S1.6（仅计划），等待用户选择；未commit/push。

## S1.6 Engineering — 2026-10-18

用户授权“继续推进S1.6学习”。启动pwd正确，保留S1.5学习确认的四份文档修改。
读取AGENTS/README/handoff/roadmap/S1.5包、S1.2文本实现与requirements；无现成S1.6包。
新增[学习包](vla/s1_6_language_generalization.md)、[script](../scripts/s1_6_language_generalization.py)。
只当前任务，不修复S1.5或启动S1.7；不改旧源码/sibling repo。执行前确认repo .venv Python3.14.7、
NumPy2.5.4，actual path在.venv/lib/python3.14/site-packages/numpy/__init__.py。无安装/数据或weights下载。
requirements只改覆盖任务注释；PNG复用stdlib helper，不依赖Pillow。

独立42参数text-only learned embedding(8,4)+masked mean+2class head，由4句/800updates/seed7/Adam lr.03训练。
train pick red/blue cube、move red/blue object；train-only vocab PAD/UNK/blue/cube/move/object/pick/red。
probe预先固定，seen4、新known-word组合4、OOV verb/noun4、OOV color4、order stress2。
除seen外规范化完整句与train交集0，未知词不加入词表；PAD/UNK零、UNK推理仍参与mean。
probe颜色别名/排除关系由评价者标注，expected class不进入decide；所有verb仅作synthetic go-to-color任务，不实现抓取。

Executed:

```bash
.venv/bin/python scripts/s1_6_language_generalization.py --output tmp/s1_6_verified
.venv/bin/python scripts/s1_6_language_generalization.py --unk-policy map --output tmp/s1_6_map_verified
.venv/bin/python scripts/s1_6_language_generalization.py --output tmp/s1_6_final
.venv/bin/python scripts/s1_6_language_generalization.py --unk-policy map --output tmp/s1_6_map_final
.venv/bin/python scripts/s1_6_language_generalization.py --output tmp/s1_6_checked
.venv/bin/python scripts/s1_6_language_generalization.py --unk-policy map --output tmp/s1_6_map_checked
.venv/bin/python -m pip check
git diff --check
```

最终checked包含exact baseline失败stage细分、tokenizer/diagnostic NPZ reload验证。train CE首步.67385→7.641e−5。
reject与map seen4/4、recombination4/4；exact lookup只有seen4/4，其他整句miss。
reject其余4/4/2句均拒绝；map OOV verb/noun4/4、OOV color2/4、order1/2。
unknown颜色同ID pairs均选red，约.5002概率；order pair同mean均选blue约.50636（舍入微差），达到碰撞上限，不是同义词/关系理解。
OOV verb/noun成功保留known颜色，不能声称理解grab/block/item；只一seed/极小人工phrase列表。

桥接6个seed41布局（复用S1.5 renderer/world，无其weights），oracle red/blue位置+固定goal-error controller，
按predicted class选目标，expected class仅评分；每接受句8actions/9states，dt.1s、world delta clip.06m、终点阈值.01m。
reject seen/recombination各24/24，OOV分别0/24、0/24、0/12（0动作/1state/truncatedFalse，仍留全分母）；
map对应24/24、12/24、6/12。错误分类stage=language_target_selection，拒绝tokenization_rejected；
exact baseline未见整句stage=exact_instruction_lookup_miss；oracle expected-class参考每组全成功。
桥接是语言正确率的六场景重复，不是独立learned视觉/控制证据，不冒充S1.5 VLA修复；S1.5仍0/16。
局部text推理每unique phrase一次，组内2或4samples median/P95，拒绝组null；不含bridge，不能称性能benchmark。

全42参数finite difference最大差7.10e−12、PAD NaN不变、OOV/mean order collision、phrase/vocab隔离、5拒绝输入通过。
weights/tokenizer/diagnostic NPZ重载通过；独立inline跨policy weights/vocab/history/diagnostics逐值相同，
所有桥接的拒绝无动作/8步执行/物理加法/限幅/成功阈值/全分母/oracle可达通过。
已查看scenes.png六layout红蓝位置；Markdown本地链接、pip check、git diff --check通过，artifacts/.venv ignored，无commit/push。
限制：人工四train句/14held-out句、mean表示、oracle几何与控制器、没有语法/真实技能或VLA泛化证据。
Current Task S1.6 Engineering Complete，Learning未勾选；Next S1.7仅计划。
用户下一步Run→先预测reject/map分组结果与失败stage→Modify→五问Explain，STOP等待用户。

## S1.6 learning confirmation — 2026-10-18

用户明确确认实验和预测均已完成，五项 Explain 准确覆盖train-only8词表、42参数、text到class的数据流、
shape/dtype和标签不入推理；区分seen/新词组合/OOV verb-noun/OOV color/order stress及held-out14句隔离。
正确手算grab red block映射IDs[1,7,1,0,…]、mean=E[red]/3，与删除词不同；4/4不证明未知词或抓取语义。
准确解释unknown-color两对ID碰撞上界2/4、order词多重集碰撞上界1/2，增加训练不能恢复已丢信息；
新增独立颜色IDs仍需语义监督，新增instead/of不能补回mean丢失的顺序关系。
正确说明reject/map不改data/vocab/weights/train loss/known预测，区分拒绝、错误选目标与控制终点错误stage。
用户明确oracle bridge24/24仅重复四个正确类别到六layout，不代表视觉/动作学习或S1.5修复，未过称robot泛化。
README和学习包Run/Modify/Explain全部勾选，roadmap同步Learning Mastered；S1.5控制0/16结论不变。
启动pwd正确，git status --short为空；本轮仅改四份状态文档，未重跑实验、安装或下载。
环境/实验沿用上节（Python3.14.7 / NumPy2.5.4），无新增能力声明。
验证命令：`git diff --check`（通过）；`git diff --stat`、`git status --short`（仅四份文档）。
Current Task = S1.6（已掌握）；Next Task = S1.7（仅计划），等待用户选择；未commit/push。

## S1.7 Engineering — 2026-10-18

用户授权“继续推进S1.7”。启动pwd正确，保留S1.6学习确认四份文档修改，读取AGENTS/README/handoff/
roadmap/S1.6包、S1.5代码与checkpoint文件结构。新增[学习包](vla/s1_7_closed_loop_vla_rollout.md)、
[script](../scripts/s1_7_closed_loop_evaluation.py)。只评价，不重训/修复、不进入Stage2，不改旧源码/sibling repo。
执行前确认repo .venv Python3.14.7 / NumPy2.5.4，actual path在.venv/lib/python3.14/site-packages/numpy/__init__.py。
无安装/数据或weights下载；requirements只改task注释，PNG用已有stdlib helper。

加载tmp/s1_5_final frozen2454参数模型，train seed7/2400steps；内存readonly，五文件SHA256前后核对，七源码hash记录。
weights SHA256=2538a3fff8fad49ac29f8599f98ebfb77ba473a29ab3ca21234a9c157fba5397。
沿用旧8validation布局×2language，不称新test。四条件nominal/kick第2步+.12m/shifted start[.18,.12]/unsupported grab red-blue block。
每策略每条件16attempts，learned/oracle/zero共192attempts/run；预算8、dt.1s、clip.06m、末maxgoalerror<=.01m。
S1.6 bridge不接入；oracle目标特权、zero忽略语言，unsupported oracle成功不证明理解指令。

Executed:

```bash
.venv/bin/python scripts/s1_7_closed_loop_evaluation.py --output tmp/s1_7_verified
.venv/bin/python scripts/s1_7_closed_loop_evaluation.py --execute 3 --output tmp/s1_7_e3_verified
.venv/bin/python scripts/s1_7_closed_loop_evaluation.py --output tmp/s1_7_final
.venv/bin/python scripts/s1_7_closed_loop_evaluation.py --execute 3 --output tmp/s1_7_e3_final
.venv/bin/python -m pip check
git diff --check
```

最终版本增加source report hash/训练元信息与全部相关本地源码hash；未改变policy或protocol数值。
Learned E1/E3前三组各0/16，完成8步但final_goal_error_after_budget；unsupported16/16拒绝，0commands/1state，
stage=policy_input_rejection，MSE null、attempts仍16；不伪造raw T0。Oracle每组16/16，zero每组0/16。
每run176条有效embodied_episode/1.0 NPZ+JSON通过validate/round-trip，16拒绝仅report；kick显式provenance，不重标action。
Offline128anchors/336valid slots/672scalars：raw5.66041e−4、clip5.64488e−4、zero6.01190e−4m²。
Learned visited action MSE nominal E1/E3=.00452874/.00454267m²，kick=.00468196/.00377318，
shifted=.00417287/.00394276；每完整组128actual actions/256scalars，分布/slots不同，不可与offline差额全归因于shift。
Oracle E3 kick visited MSE.0001828125m²但终点error0/success16，展示stale suffix动作误差与后续恢复不矛盾。
完整trace/commands/evaluator expert/decision prefix/error/ever_goal/stage/timing保留；stage为症状非模态根因。

正式计时前5warm-up（检查额外调用亦排除）；记录render/policy/apply_prefix/local evaluator loop median/P95/max/count，
all-attempt与accepted分开，unsupported accepted null。例final E1 nominal policy median/P95 .1271/.1724ms（128calls），
E3 .1435/.1889ms（48calls）；本机调度相关，不作性能承诺。
本地loop含诊断/内存日志，不含raw导出/sensor/transport/plant/wait；sim timestamp与wall compute分开，无hard real-time结论。

真实unsupported拒绝边界、反事实evaluator goal改变但learned commands/trace不变、4坏chunk拒绝通过。
独立inline验证192attempts、176raw+16拒绝、source/跨Eoffline/hash不变，逐条检查raw schema、T/T+1、
物理kick/加法、prefix、clip、实际state expert MSE、成功阈值、全分母；nominal/kick learned trace与S1.5旧报告逐值相同。
已查看scenes.png四布局；Markdown links、pip check、git diff --check通过；artifacts/.venv ignored，无commit/push。
限制：单checkpoint、旧validation、静态RGB/ideal point、固定kick/新起点；未修复learned VLA，不声称真实robot/泛化或实时性。
Current Task S1.7 Engineering Complete，Learning未勾选；Stage1工程包全部完成但控制能力仍失败。
Next S2.1仅计划；用户下一步Run→预测E1/E3指标与反馈→Modify→五问Explain，STOP，不自动进入Stage2。

## S1.7 learning confirmation — 2026-10-18

用户明确确认实验和预测均已完成，五项 Explain 准确覆盖冻结policy/词表/契约/split/hash/协议与四组case，
真实目标与counterfactual expert仅供evaluator，评价中重训或接oracle bridge会改变被评对象。
准确说明expert-state offline128anchors/336slots/672scalars与visited每完整条件256scalars的状态/动作槽/权重差异。
正确区分三个failure stage，拒绝保持16attempts而0action/MSE null、不造T0 raw episode。
准确描述E1/E3保持权重/数据/目标/offline/预算，calls8→3、prefix3/3/2、kick后tick2/tick3反馈；
oracle E3 disturbed MSE.0001828125m²与终点16/16不矛盾，动作误差不等于终点成功。
正确指出local_loop包含render/policy/apply与诊断，不能与子阶段相加；sim timestamp不等于wall time。
明确局部overrun0、oracle全成功、hash不变和Stage1工程完成均不足证明实时性/learned能力/模型正确/真实robot泛化。
README/学习包Run/Modify/Explain全部勾选，roadmap同步S1.7与Stage1 Learning Mastered；S1.5/S1.7策略控制失败结论不变。
启动pwd正确，git status --short为空；本轮只改四份状态文档，未重跑实验、安装或下载。
环境/工程证据沿用上节（Python3.14.7 / NumPy2.5.4），不增加能力声明。
验证命令：`git diff --check`（通过）；`git diff --stat`、`git status --short`（仅四份文档）。
Current Task = S1.7（已掌握）；Stage1工程与学习包全部完成、策略能力仍未达标。
Next Task = S2.1（仅计划），等待用户选择，不自动进入Stage2；未commit/push。


## S2.1 Engineering — 2026-10-18

用户明确请求继续S2.1学习；启动pwd正确、git status --short为空。
读取AGENTS/README/handoff/roadmap及S0.1模板；此前没有S2.1包。
新增[学习包](vla/s2_1_rt_overview.md)和[script](../scripts/s2_1_rt_overview.py)。
只推进S2.1，不修复S1.5、不进入Octo、不改sibling repo。
官方来源：RT-1 arXiv2212.06817v1、RT-2 arXiv2307.15818v1、两个官方project站；
RT-1 git ls-remote确认HEAD 4569641b8111f3f402c32d8e24becd2a6e952ecc，浏览固定revision的
transformer_network.py/action_tokenizer.py；未执行TF或验证RT-2完整源码。
区分论文视觉摘要48tokens与完整动作槽、当前动作字段与未来chunk、token约束与语义迁移。

执行前验证repo .venv Python3.14.7 / NumPy2.5.4，实际NumPy路径为
.venv/lib/python3.14/site-packages/numpy/__init__.py。无安装/公开dataset或weights下载。
复用S0.5三train episode、12raw命令拟合S0.6 ranges；取首episode4×2float32 joint absolute rad命令。
channel offset ID与共享bin有序字符串往返一致、中心解码半bin界、6坏字符串拒绝、合成logits词表限制通过。
两次输出NPZ raw/decoded重载一致；无learned模型、闭环success或模型latency声明。

Executed:

```bash
.venv/bin/python scripts/s2_1_rt_overview.py --output tmp/s2_1_verified
.venv/bin/python scripts/s2_1_rt_overview.py --bins 16 --output tmp/s2_1_b16_verified
.venv/bin/python -m pip check
git diff --check
```

B256/B16均PASS；Markdown本地links及跨B raw/range/shape不变量独立检查通过。
产物ignored tmp/；README为状态源，学习三项保持未勾选；无commit/push。
限制：只有codec/格式边界证据，不是RT架构前向/预训练迁移/真实robot复现。
Current Task=S2.1 Engineering Complete，Learning pending；Next Task=S2.2仅计划。
下一步用户阅读学习包、Run、预测B变化后Modify、回答五项Explain；STOP等待用户。


## S2.1 learning confirmation — 2026-10-18

用户明确确认实验和预测均已完成，五项Explain准确覆盖：
RT-1 sentence embedding/FiLM视觉条件化与robot监督，RT-2 VLM和web/robot co-fine-tuning；
48视觉tokens、11/8当前动作字段与未来horizon是不同轴；bin ID、数字字符串与内部token ID不同，
PaLI-X整数token和PaLM-E低频token映射、本地parser不等于VLM tokenizer。
正确说明B256→16保持raw/shape/dtype/range/字段/H1/rad语义，channel空间512→32、半bin界扩大16倍，
实际误差不必同比变化且界仅对in-range成立。
准确区分合法token、语义迁移、新运动技能与真实robot success；明确本地只有codec/量化/词表证据。
README与学习包Run/Modify/Explain全部勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅更新四份状态文档，未运行Python实验、安装或下载。
环境/实验沿用S2.1工程记录（Python3.14.7 / NumPy2.5.4），未新增模型或能力声明。
验证：git diff --check通过；git diff --stat与git status --short核对仅四份文档。
Current Task=S2.1（已掌握）；Next Task=S2.2 — Octo Study（仅计划），等待用户选择；未commit/push。


## S2.2 Engineering — 2026-10-18

用户授权继续S2.2。启动pwd正确，保留S2.1学习确认的四份文档修改。
读取AGENTS/README/handoff/roadmap/S0.1模板；此前无S2.2 package。
新增[学习包](vla/s2_2_octo_study.md)、[script](../scripts/s2_2_octo_trace.py)。仅推进S2.2。
官方来源：Octo项目与arXiv2405.12213v1；git ls-remote核对HEAD
241fb3514b7c40957a86d869fecb7c7fc353f540，读取固定octo_module.py、action_heads.py、
scripts/configs/octo_pretrain_config.py、octo_model.py。前三份小源码保存ignored tmp/s2_2_source供阅读，非dataset/weights。
猜测examples/finetune_new_observation_action.py返回404，没有以其作为适配证据。
区分论文/HEAD配置；选MLP DiffusionActionHead，不混UNet。官方W2/H4/A7、repeat task、goal stacking、
readout因果同组访问、sample_actions反归一化与全模型微调均在文档解释；未运行JAX/Flax/官方checkpoint。

执行前核对repo .venv Python3.14.7 / NumPy2.5.4，实际模块
.venv/lib/python3.14/site-packages/numpy/__init__.py。requirements不变，无安装/数据或weights下载。
本地7×4随机embedding、两层identity-QKV residual attention、单task prefix，非完整Transformer。
两条件present/missing有效tokens7/5、allowed edges28/14；未来隔离、readout不影响obs/task、
task改变readout、masked weights0、missing NaN不影响输出、valid NaN拒绝均PASS。
同一readout随机head输出(3,2)/(3,3)，仅验证shape适配，不是learned动作。
独立物理fixture(1,2,3,2)float32 worldXY delta[m]/dt.1s，以固定.06m单位尺度归一化，
alpha_bar.64+seed11噪声后oracle反解误差<=1e-7；不训练/采样denoiser，不导出伪episode。

Executed:

```bash
.venv/bin/python scripts/s2_2_octo_trace.py --output tmp/s2_2_verified
.venv/bin/python scripts/s2_2_octo_trace.py --wrist missing --output tmp/s2_2_missing_verified
.venv/bin/python -m pip check
git diff --check
```

两run八类checks与逐数组NPZ重载PASS；独立inline检查mask精确key集合、跨flag全部数组不变、
cases相同/selected不同、扩散物理往返，Markdown本地links通过。pip check无冲突。
无learned policy/控制/语义迁移/latency或论文复现声明，S1.5闭环失败不变。
产物ignored tmp/，无commit/push。README/roadmap同步S2.2工程完成，学习三项未勾选。
Current Task=S2.2；Next Task=S2.3仅计划。用户下一步Run→预测wrist模式后Modify→五问Explain；STOP。


## S2.2 learning confirmation — 2026-10-18

用户明确确认实验和预测均完成，五项Explain准确覆盖Octo条件输入/分组mask/Transformer/readout/
conditional diffusion/反归一化到执行，与RT-2离散自回归路径区别；goal为条件、readout非动作。
准确列出present/missing primary0与readout1可读key集合，shape7×4保持、valid7→5、edges28→14；
明确mask隔离不补全感知。正确区分本地W2/H3/A2与官方W2/H4/A7、20去噪步及k非物理时刻。
正确解释cases每次都算、flag只选selected，mask改变softmax和readout；invalid NaN需先清理、valid NaN拒绝。
准确区分A2→A3 shape适配与sensor/action/unit/frame/controller/统计/训练/微调及闭环能力；
oracle知道epsilon的0.8a+0.6epsilon反解仅验证代数，真实策略需学习条件预测并反向采样。
用户明确真实成功仅限对应测试条件，不外推任意技能、跨机器人泛化或安全实时性。
README与学习包Run/Modify/Explain勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅更新四份状态文档，未重跑实验、安装或下载。
环境/工程证据沿用上节Python3.14.7 / NumPy2.5.4，不增加模型或控制能力声明。
验证：git diff --check通过；git diff --stat、git status --short核对仅四份文档。
Current Task=S2.2（已掌握）；Next Task=S2.3 — OpenVLA Architecture Trace（仅计划）。等待用户选择；未commit/push。


## S2.3 Engineering — 2026-10-18

用户授权继续S2.3。启动pwd正确、git status --short为空；读取AGENTS/README/handoff/roadmap及模板结构。
新增[学习包](vla/s2_3_openvla_architecture_trace.md)、[来源manifest](vla/s2_3_openvla_sources.json)、
[script](../scripts/s2_3_openvla_trace.py)。仅原版OpenVLA-7B，不引入OFT或推进S2.4。
官方paper arXiv2406.09246v1/project；git ls-remote HEAD核对
c8f03f48af692657d3060c19588038c7220e9af9，读固定HF modeling/processing、action_tokenizer、
datasets、data_utils及DINO/SigLIP vision代码。timm v0.9.10/LlamaConfig v4.40.1只作静态维度查阅。
HF API metadata revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f；web未获取HF config，
通过只读HTTP取config60712bytes/preprocessor1650bytes及小源码，均ignored tmp/s2_3_source；无权重/tokenizer或真实dataset。
manifest保留revision/config subset/源码SHA256，不导入上游模型，不安装Torch/timm/transformers。
原版load_proprio=False、W1/H1；unnorm_key选择统计及动作维数、不作为LLM条件；controller state与policy输入区别已写明。
静态shape追踪双分支(256,1024)/(256,1152)沿feature拼接→2176→projector4096→BOS后插入。
区分论文2层projector概述与固定HF fused3层；词表32064−64=32000、256grid points/255centers，末端两ID同中心。

执行前核对repo .venv Python3.14.7 / NumPy2.5.4，actual module位于
.venv/lib/python3.14/site-packages/numpy/__init__.py；requirements未改，无环境修复。
本地16×16 RGB、16平均patches、3/5维随机proxy、单线性projector和8维identity-QKV causal attention；
没有真实encoder/tokenizer/LLM、训练或生成；固定teacher-forced action IDs仅作序列/codec边界。
prefix20/teacher23槽，labels20/21由输出19/20预测，未来字段不影响之前embedding，NPZ逐值重载通过。
101条synthetic train域worldXY delta[m]数值分别拟合compact/wide q01/q99，raw保存；同ID/normalized，wide物理数值2倍。
strict拒绝3类坏ID；官方permissive text ID1也解成边缘数值，展示合法数值不证明合法token。
隔离执行已读官方ActionTokenizer class，dummy tokenizer只回IDs，无transformers/tokenizer模型执行；
本地encode1256inputs、decode全部256IDs与官方逐值等价。没有真实文本tokenizer roundtrip证据。

Executed:

```bash
.venv/bin/python scripts/s2_3_openvla_trace.py --output tmp/s2_3_verified
.venv/bin/python scripts/s2_3_openvla_trace.py --unnorm-key wide --output tmp/s2_3_wide_verified
.venv/bin/python scripts/s2_3_openvla_trace.py --output tmp/s2_3_final
.venv/bin/python scripts/s2_3_openvla_trace.py --unnorm-key wide --output tmp/s2_3_wide_final
.venv/bin/python -m pip check
git diff --check
```

最终两run加入raw train数值保存，其余计算不变；序列/causal/shift/codec端点/255中心/统计比例/mask检查PASS。
独立inline检查所有跨key NPZ数组相同、q统计仅来自保存train、原始raw保持、label前缀遮罩、feature拼接及物理解码通过。
Markdown本地links、来源manifest JSON、pip check、git diff --check通过，artifacts ignored。
无模型loss、闭环success/latency或真实OpenVLA复现声明；未修复S1.5，未修改sibling repo，无commit/push。
README/roadmap工程状态已同步，Learning三项未勾选。Current Task=S2.3；Next Task=S2.4仅计划。
用户下一步Run→预测统计key变化再Modify→五问Explain，STOP。


## S2.3 learning confirmation — 2026-10-18

用户明确确认实验和预测完成，五项Explain准确覆盖原版单图/image+text输入、DINOv2/SigLIP/projector/
Llama2/action codec/统计/controller路径；load_proprio=False，unnorm_key选统计和A而非state token，
低层controller仍可使用实际state。正确区分(B,6,224,224)两分支预处理与双camera/时间，
256patch沿feature拼1024+1152=2176，HF fused projector2176→8704→4096→4096；本地随机代理非原模型。
准确定位23槽中BOS0/patch1–16/text17–19/action20–21/EOS22，输出19/20预测label20/21；
teacher forcing非生成，EOS非robot terminate。
正确解释32064−64=32000、256grid points/255centers、ID31744–31999、末端31745/31744碰撞，
以及非法text ID1被宽松解码为边缘数值不证明合法性。
准确解释compact→wide不改输入/features/序列/ID/normalized/H1/A2，统计与selected物理数值变2倍，
flag仅改selected且trace相同；真实key若改变A也改变生成长度，统计正确不等于语义/frame/控制成功。
README与学习包Run/Modify/Explain全部勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅更新四份状态文档，未重跑实验、安装或下载。
环境/工程证据沿用上节Python3.14.7 / NumPy2.5.4，不增加真实OpenVLA/robot能力声明。
验证：git diff --check通过；git diff --stat、git status --short核对仅四份文档。
Current Task=S2.3（已掌握）；Next Task=S2.4 — FAST / π0-FAST Action Representation（仅计划）。
等待用户选择；未commit/push。


## S2.4 Engineering — 2026-10-18

用户授权继续S2.4。启动pwd正确、git status --short为空；读AGENTS/README/handoff/roadmap，当前包此前不存在。
新增[学习包](vla/s2_4_fast_action_representation.md)、[source manifest](vla/s2_4_fast_sources.json)、
[script](../scripts/s2_4_fast_tokenization.py)。仅FAST/π0-FAST表示，不进入S2.5或修改旧控制模型。
官方FAST arXiv2501.09747v1、pi.website/research/fast；git ls-remote核对openpi HEAD
215abfb217dbac7d5f1273282331b9b1866c0479，读tokenizer.py/pi0_fast.py。
HF API固定physical-intelligence/fast revision ec4d7aa71691cac0b8bed6942be45684db2110f4，
仅取processing_action_tokenizer.py6145bytes/processor_config.json253bytes/API metadata。
最初preprocessor_config.json为404，依据API列表更正文件名。源码/metadata存ignored tmp/s2_4_source，
未下载发布BPE字典tokenizer.json、策略权重或真实dataset，未执行trust_remote_code。
文档区分FAST/FAST+/π0-FAST、state文本prefix与动作FAST IDs、PaliGemma映射跳过128 special，
官方Unicode偏移/低域clip/坏decode零fallback、scale10/vocab2048与类默认1024区别。

执行前核对repo .venv Python3.14.7 / NumPy2.5.4，actual module为
.venv/lib/python3.14/site-packages/numpy/__init__.py；requirements未改，无安装/环境修复。
NumPy显式正交DCT-II沿H16时间轴、A2worldXY delta[m]、dt.1s，固定.06m归一化；不拟合验证统计。
8个synthetic train chunks拟合整数pair BPE（固定基础字母−128…128，最多24 merges），
smooth/seed31 rough两个heldout独立chunk仅评价。raw/coeff/Q/IDs/decoded/baseline全部保存，不伪造episode。
本地不是Unicode/byte-level FAST+、没有VLA或动作生成；BPE整数无损但量化有损。

Executed:

```bash
.venv/bin/python scripts/s2_4_fast_tokenization.py --output tmp/s2_4_verified
.venv/bin/python scripts/s2_4_fast_tokenization.py --scale 20 --output tmp/s2_4_s20_verified
.venv/bin/python -m pip check
git diff --check
```

两run正交/逆DCT/BPE整数往返、RMSE上界、6类拒绝、JSON字典/NPZ重载PASS。
scale10/20各10/9merges、vocab267/266；32个系数槽不变。
smooth tokens2→6，RMSE均.0002436865934m、积分位移误差.0048m；rough均32tokens，
RMSE.0016901615374→.0009441706822m、积分偏差.0077020950→.0049056572m。
scalar bins256固定32tokens，smooth/rough RMSE.0001440436/.0001385139m；不声称精度匹配对照。
理论RMSE上界.003→.0015m，不是单component最大误差界，也不保证实际误差减半。
积分是ideal additive point数值诊断，不是真实闭环success；N不含字典/prompt/framing，不是bytes或速度倍数。

独立inline检查跨scale raw/连续DCT/scalar baseline不变量、正交能量、DC决定积分偏差、
重载merge重编码与整数恢复、train-only refit字典、manifest JSON及Markdown本地链接通过。
pip check无冲突、git diff --check通过。无policy loss/robot success/latency或官方提速复现声明。
所有产物ignored tmp/，无commit/push。README/roadmap同步工程完成，学习三项未勾选。
Current Task=S2.4；Next Task=S2.5仅计划。用户Run→先预测scale再Modify→五问Explain，STOP。


## S2.4 learning confirmation — 2026-10-18

用户明确确认实验和预测完成，五项Explain准确区分FAST方法、FAST+通用词表、π0-FAST自回归策略；
训练chunk编码为监督标签，推理生成token再解码，robot state属于策略条件而非动作codec主体。
正确描述H16/A2在DCT后时间轴变频率轴、flatten32与变长N，不将N=2当成两个物理控制步。
明确无裁剪归一化/DCT/flatten可逆、BPE对量化整数无损、round和clipping有损，RMSE小仍可积分偏移。
准确解释scale10→20保持raw/H/A/dt/尺度/DCT连续系数，Q变化后同train重fit字典；
理论RMSE界.003→.0015m不保证实际误差减半，smooth误差不变而token2→6展示压缩取舍。
正确说明完整train chunks拟合、heldout不调字典/字母域，rough32→32是未压缩而非decode失败；
坏流拒绝仅验证契约，token数/bytes/policy latency/robot success不能互相替代，本地只有codec证据。
README与学习包Run/Modify/Explain勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅更新四份状态文档，未重跑实验、安装或下载。
环境/工程证据沿用上节Python3.14.7 / NumPy2.5.4，不增加FAST+或VLA控制复现声明。
验证：git diff --check通过；git diff --stat、git status --short核对仅四份文档。
Current Task=S2.4（已掌握）；Next Task=S2.5 — π0 Architecture Trace（仅计划），等待用户选择；未commit/push。


## S2.5 Engineering — 2026-10-18

用户授权继续S2.5。启动pwd正确，保留S2.4学习确认四份文档修改；读AGENTS/README/handoff/roadmap。
新增[学习包](vla/s2_5_pi0_architecture_trace.md)、[manifest](vla/s2_5_pi0_sources.json)、
[script](../scripts/s2_5_pi0_flow_trace.py)。当前仅原版π0，不进入π0.5。
官方paper arXiv2410.24164v1、blog/pi0（research/pi0不可访问，沿论文blog链接跳转）；
git ls-remote确认openpi HEAD215abfb217dbac7d5f1273282331b9b1866c0479。
固定读取pi0.py/pi0_config.py/gemma.py，保存ignored tmp/s2_5_source与manifest hash。
明确选择JAX Pi0Config(pi05=False)，state连续suffix、time与action拼接MLP，非pi05 adaRMS。
静态W_vlm2048/W_expert1024经独立投影共用attention；H50/A_pad32与论文实际最大18分开。
代码t1noise→t0action、x=t noise+(1−t)action、u=noise−action、Euler负步长，与论文τ方向配套转换；
记录v1印刷公式表面符号不一致，不混用来实现。未执行JAX/Flax/Torch、未下载模型/数据。

执行前核对repo .venv Python3.14.7 / NumPy2.5.4，actual module
.venv/lib/python3.14/site-packages/numpy/__init__.py；无安装/修复，requirements不变。
本地prefix3×6/suffix4×4（state1/action3）、共同QKV宽3、两层随机attention；mask34边，
full/cache差0、prefix不受suffix影响、state不受action影响、state影响action通过。
独立raw3×2 worldXY delta[m]/dt.1s，.06m固定尺度、seed29噪声，非episode/rollout。
常量oracle需特权clean/noise，K10/20 normalized RMSE5.78e−17/8.92e−17；biased oracle +.1均RMSE.1。
解析场v=x−mu的精确终点mu+(noise−mu)e^(−1)，不是clean；Euler误差.00927026136→.00453519969。
无learned velocity、策略训练或实际控制，因此不报告policy loss/robot success/latency。

Executed:

```bash
.venv/bin/python scripts/s2_5_pi0_flow_trace.py --output tmp/s2_5_verified
.venv/bin/python scripts/s2_5_pi0_flow_trace.py --steps 20 --output tmp/s2_5_k20_verified
.venv/bin/python -m pip check
git diff --check
```

两run三类拒绝/NPZ重载通过；独立inline检查mask精确key集合、固定prefix多suffix缓存等价、
跨K全部固定输入不变、负时间网格、oracle/biased终点、解析Euler封闭式与误差改善通过。
隔离已读官方make_attn_mask函数以NumPy代替jnp，mask与本地逐值相同；没有导入上游依赖/运行模型。
Markdown links、manifest JSON、pip check与git diff --check通过，产物ignored tmp/，无commit/push。
限制：本地attention与ODE是分开的无训练探针，不是flow策略；未修复S1.5、不改sibling repo。
Current Task=S2.5 Engineering Complete，Learning三项未勾选；Next Task=S2.6仅计划。
用户Run→先预测K变化再Modify→五问Explain，STOP。


## S2.5 learning confirmation — 2026-10-18

用户明确确认实验和预测完成，五项Explain准确覆盖原版VLM image/text prefix、连续state suffix、
noisy action/time输入和velocity输出，区分π0-FAST离散自回归与pi05条件机制。
正确解释2048/1024各自QKV投影到共同attention后再映回原宽度；H50/A_pad32/state1与实际DoF区别，
本地prefix3/state1/action3的mask34边及双向action访问。
准确写出代码x_t=tε+(1−t)a、u=ε−a、t1→0/负步长；换τ=1−t时velocity变a−ε、正步长0→1。
正确解释prefix不读suffix可缓存，当前实现state在suffix每步重算不等于理论上不能缓存；
新image/text/mask使旧cache失效，full/cache等价仅证明本地固定条件下数值路径一致。
准确区分K与H/A/robot dt，K10→20的trace11→21及三场结果：oracle浮点误差、biased .1不消失、
curved Euler对解析终点误差.00927026→.00453520；明确均非learned策略/闭环/latency证据。
README与学习包Run/Modify/Explain勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅更新四份状态文档，未重跑实验、安装或下载。
环境/工程证据沿用上节Python3.14.7 / NumPy2.5.4，不增加真实π0/robot能力声明。
验证：git diff --check通过；git diff --stat、git status --short核对仅四份文档。
Current Task=S2.5（已掌握）；Next Task=S2.6 — π0.5 Study（仅计划），等待用户选择；未commit/push。

## S2.6 Engineering — 2026-10-19

用户明确请求“推进S2.6的学习”，仅推进本任务。启动pwd正确，git status --short为空。
读取AGENTS、README、handoff、roadmap、S0.1模板、S2.4/S2.5包；此前无S2.6包。
新增[学习包](vla/s2_6_pi05_study.md)、[manifest](vla/s2_6_pi05_sources.json)、
[CPU探针](../scripts/s2_6_pi05_conditions.py)。
浏览官方arXiv2504.16054v1与pi.website/blog/pi05；固定沿用S2.5 openpi revision
215abfb217dbac7d5f1273282331b9b1866c0479（非当前HEAD声明）。curl -fsSL读取该revision的
src/openpi/models/{pi0.py,pi0_config.py,tokenizer.py,gemma.py}至ignored tmp/s2_6_source，hash写入manifest。
静态选择JAX Pi0Config(pi05=True)：state文本prefix，suffix仅H动作槽，time MLP→adaRMS。
区分论文完整异构训练/联合FAST CE与flow/高层子任务系统和所选pi0.py单flow loss实现。
不执行上游依赖、不下载词表/权重/数据、不修改旧repo、不修复S1.5策略。

本轮实际环境与历史S2.5不同：repo .venv Python3.12.3，NumPy2.5.4，
解释器/home/lucas/projects/embodied-ai-playground/.venv/bin/python，sys.prefix为repo .venv，
actual NumPy .venv/lib/python3.12/site-packages/numpy/__init__.py；执行前已核对，无安装或修复。
requirements未变。默认Conda python仅查看路径，未用于运行实验或安装。

Executed:

```bash
.venv/bin/python scripts/s2_6_pi05_conditions.py --output tmp/s2_6_verified
.venv/bin/python scripts/s2_6_pi05_conditions.py --delta .02 --output tmp/s2_6_delta20_verified
.venv/bin/python -m pip check
git diff --check
```

当前world XY位置[m]，固定1m教学尺度，无fit：1mm改变bins仍[128,128]，20mm为[130,128]。
上游表达式边界[-1.01,-1,0,1,1.01]→[-1,0,128,255,255]，没有双边显式clip。
hybrid教学mask prefix3/discrete2/continuous3共36边，两层传播FAST代理标签扰动对continuous差0；
错误开放continuous读取discrete边，差67.5。无模态语义的8×4 synthetic float64输入。
adaRMS零初始化条件差0，seed41非零modulation差.2635350535385911；不是trained参数或完整Transformer。
每run3类非法state拒绝、NPZ逐数组重载通过。
独立inline检查隔离提取上游PaligemmaTokenizer.tokenize纯digitize AST表达式与本地相同，
精确mask集合、两run固定数组不变、delta0/负/NaN拒绝、已有目录拒绝、manifest JSON及Markdown本地链接通过。
pip check无冲突，git diff --check通过；所有运行产物ignored tmp/。未commit/push。

限制：无learned模型、heldout策略评价、robot success/failure stages/latency；只能验证局部信息路径。
README/roadmap更新Engineering Complete；Run/Modify/Explain保持未勾选，仅用户报告可确认学习。
Current Task=S2.6；Next Task=S2.7仅计划。用户读学习包→Run→先预测delta再Modify→五问Explain；STOP。


## S2.6 learning confirmation — 2026-10-19

用户明确确认实验和预测完成；五项Explain准确区分π0连续state suffix/action-time融合、
π0-FAST离散state prefix与AR动作、π0.5 state prefix/连续flow/逐层adaRMS。
明确pi05开关不包含完整论文的数据配方、联合训练、层级推理与部署。
正确给出p(a,s|o,l)=p(a|o,s)p(s|o,l)，区分高低层条件、共享模型和刷新频率，
以及离散预训练/联合监督与最终flow动作生成。反向时间积分指本包选定代码约定。
正确指出FAST目标既能直接泄漏，也能经prefix多层间接泄漏，需同时隔离。
准确解释固定1m尺度下bin宽7.8125mm、1mm与20mm对应[128,128]/[130,128]，
bin不是SentencePiece ID，state更新需刷新受影响prefix KV。
准确区分adaRMS零初始化与非零条件调制、随机参数与训练证据，及进度分/选物率/最终成功率。
README与学习包Run/Modify/Explain勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅更新四份状态文档，未重跑实验、安装或下载。
运行环境与实验结果沿用上节Python3.12.3 / NumPy2.5.4；没有新增机器人泛化或性能结论。
验证：git diff --check通过；git diff --stat与git status --short核对仅四份文档。
Current Task=S2.6（已掌握）；Next Task=S2.7 — VLA Architecture Comparison，仅计划。
等待用户选择，不自动进入下一任务；未commit/push。

## S2.7 Engineering — 2026-10-19

用户明确请求继续S2.7；启动pwd正确，已有S2.6学习确认的四份文档修改全部保留。
读取AGENTS、README、handoff、roadmap、S0.1模板及S2.1–S2.6学习包；此前无S2.7包。
新增[统一对照学习包](vla/s2_7_vla_architecture_comparison.md)与[预算脚本](../scripts/s2_7_architecture_budget.py)。
本轮浏览Octo/OpenVLA官方项目、FAST官方介绍与paper v1、π0 paper v1、π0.5官方介绍；
源码版本沿用既有包：Octo241fb3514b7c40957a86d869fecb7c7fc353f540，
OpenVLA c8f03f48af692657d3060c19588038c7220e9af9，openpi215abfb217dbac7d5f1273282331b9b1866c0479。
GitHub tree浏览返回Internal Error，精确实现证据沿用既有固定源码审计/manifest，未声称重新执行上游。
统一模态/fusion/action/distribution/chunk/data/adaptation/fine-tune/deployment比较，
强调AR也是generative、FAST codec≠policy、原版OpenVLA H1、pi05分支≠完整论文系统。

执行前核对repo .venv解释器与prefix，Python3.12.3；已有NumPy2.5.4实际模块
.venv/lib/python3.12/site-packages/numpy/__init__.py。新脚本仅标准库，无安装、数据/权重下载。
假设H4/A7/N8/dt.1s；prefix40ms/token5ms/expert4ms/smallhead.5ms/io5ms全部人为设定。
四路径为教学scalar chunk AR、FAST chunk AR、flow expert、small diffusion head，不是实际模型测速。
默认E1/K10串行次数28/8/10/20，假设耗时185/85/85/55ms，预算100ms，margin−85/15/15/45ms。
E4保持计算路径和耗时，预算400ms、margin215/315/315/345ms；K20仅flow变125ms、E1 margin−25ms。
E·dt只是执行/计算重叠的稳态预算，启动延迟/观测新鲜度/开环风险另需评价，不能当真实实时或成功证据。

Executed:

```bash
.venv/bin/python scripts/s2_7_architecture_budget.py --output tmp/s2_7_verified
.venv/bin/python scripts/s2_7_architecture_budget.py --execute 4 --output tmp/s2_7_e4_verified
.venv/bin/python scripts/s2_7_architecture_budget.py --steps 20 --output tmp/s2_7_k20_verified
.venv/bin/python -m pip check
git diff --check
```

三run JSON重载一致。独立inline断言核对手算成本、E/K不变量、H1时7个scalar调用、
9类非法参数拒绝、已有目录FileExistsError与原report不变、三个输出重载和Markdown本地链接，均通过。
一次已有目录CLI调用非零退出是预期拒绝，随后subprocess核验异常类型与不覆盖。
pip check无冲突；git diff --check通过；生成结果ignored tmp/，requirements不变。
无真实模型forward/训练、learned评价、机器人success或latency测量；S1.5控制失败结论不变。
README/roadmap更新S2.7 Engineering Complete；学习三项未勾选，仅用户报告可确认。
Current Task=S2.7；Next Task=S3.1 — State Dynamics，仅计划；不自动开始Stage3。
用户阅读统一表→Run→先预测E变化再Modify→五问Explain。未commit/push。


## S2.7 learning confirmation — 2026-10-19

用户明确确认实验和预测完成；五项Explain准确比较Octo、原版OpenVLA、π0-FAST、π0、π0.5的
modality/fusion/action路径，区分H1/A7、FAST codec与完整policy、开源分支与论文语义层级。
正确说明AR属于generative，teacher forcing训练并行与部署串行不同，
Octo小diffusion head和π0 family带attention的expert单步成本不可按次数直接比较。
正确指出tokenizer/head/unnorm_key不足以完成语义适配，需审计frame/unit/anchor/controller/dt、
数据边界与泄漏、目标域训练，并评价visited-state、闭环全分母、失败和端到端延迟。
准确给出E1→4保持28/8/10/20调用与185/85/85/55ms假设耗时、预算100→400ms、margin均加300ms；
K10→20仅flow变125ms、E1 margin+15→−25ms，不能推出真实速度或success。
明确匹配版本/checkpoint、输入、任务/初态/扰动/训练预算/试验分母与硬件/dtype/batch/H/E/K/N，
记录warmup、p50/p95、deadline miss；论文实证、静态源码与本地probe证据不可互相替代。
README及学习包Run/Modify/Explain勾选，roadmap同步；Stage2全部Engineering Complete + Learning Mastered。
启动pwd正确，git status --short为空；本轮仅更新四份状态文档，未重跑实验、安装或下载。
环境与工程结果沿用上节Python3.12.3标准库预算计算，不增加真实机器人性能或大模型复现结论。
验证：git diff --check通过；git diff --stat、git status --short核对仅四份文档修改。
Current Task=S2.7（已掌握）；Next Task=S3.1 — State Dynamics，仅计划。
等待用户选择，不自动开始Stage3；未commit/push。

## S3.1 Engineering — 2026-10-20

用户明确请求推进S3.1学习；启动pwd正确、git status --short为空，读AGENTS、README、
roadmap、handoff与S0.1模板；此前无S3.1任务包。本轮仅S3.1，未开始S3.2。
新增[学习包](vla/s3_1_state_dynamics.md)与[脚本](../scripts/s3_1_state_dynamics.py)。
自包含schema s3_1_xy_command_dynamics/1.0，明确commanded world XY displacement[m]不同于
S0.1 realized ideal delta；位置相关gain与XY串扰，dt=.1s，40个独立reset、每个20转移/21状态。
raw float32、timestamp float64；MLP/Adam float64 CPU。训练30episodes/600转移，验证10/200，
先episode split后flatten，输入与实际位移target逐通道统计只fit train。
4→32 tanh→2，226参数；手写反向与Adam，无新依赖、安装、dataset/weights下载或旧repo访问。

执行前核对 `.venv/bin/python` / prefix，Python3.12.3、NumPy2.5.4；模块实际路径
`/home/lucas/projects/embodied-ai-playground/.venv/lib/python3.12/site-packages/numpy/__init__.py`。

Executed final code:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_1_state_dynamics.py --output tmp/s3_1_final > tmp/s3_1_final_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_1_state_dynamics.py --epochs 20 --output tmp/s3_1_short_final > tmp/s3_1_short_final_stdout.json
.venv/bin/python -m pip check
git diff --check
```

此前初版分别运行tmp/s3_1_verified、tmp/s3_1_short_verified，数值误差与最终版一致；
最终版补充机器可读contract.json。默认validation MSE1.9008076e-8m²/RMSE.137870mm，
不动17.286919mm、理想加法6.848537mm；20步MLP3.096412mm，两基线不变。
默认train RMSE.099385mm；四tensor中心差分探针最大绝对误差2.692e-11。
最终batch1推理p50/p95=.005813/.006945ms，30warmup/200samples，含标准化与decode，
不含I/O、环境、policy或planning；首轮.005935/.007450ms体现运行波动。

独立inline验证：raw shape、完整episode split无重叠、train统计重算、validation位置+100m不影响
train统计、checkpoint重载精确复现预测、物理MSE重算、2000/20步全部raw/split/统计及两个基线
不变；CLI epochs0拒绝且不创建目录、已有目录拒绝且report字节不变，全部PASS。
pip check无冲突，文档本地链接与git diff --check通过；所有实验输出ignored tmp/。
未测closed-loop success/failure stage、OOD或multi-step稳定；toy单步结果不能替代机器人成功。
README/roadmap更新Engineering Complete，Learning三项保持未勾选；未commit/push。
Current Task=S3.1；Next Task=S3.2仅计划。用户先读Problem→Architecture，再Run，
先预测epochs变化后Modify，用自己的话回答五项Explain；等待明确学习反馈，不自动推进。

## S3.1 learning confirmation — 2026-10-20

用户明确确认实验和预测均已完成，五项Explain准确覆盖：VLA policy与forward dynamics的
输入输出、完全可观测确定性XY不需要language、commanded displacement与realized displacement
及位置相关gain/XY串扰、惯性系统需速度等Markov信息。
正确给出20actions/21states时序、4维输入/2维实际位移target、反标准化加当前状态，
解释完整episode split与train-only mean/std、无量纲训练MSE和m²/MSE及m/RMSE单位。
准确指出2000→20步保持数据/split/统计/契约/验证输入目标与基线不变，改变权重/预测/loss/误差；
单步更准不证明多步稳定、policy/planning/controller或真实机器人闭环成功。
README与学习包Run/Modify/Explain已勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅修改四份状态文档，未重跑Python实验、安装或下载。
环境与实验结果沿用上述S3.1工程记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU），
不增加多步稳定、OOD或机器人成功声明。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对仅四份文档变更。
Current Task=S3.1（已掌握）；Next Task=S3.2 — Multi-step Rollout，仅计划。
等待用户选择，不自动进入下一任务；未commit/push。

## S3.2 Engineering — 2026-10-20

用户明确请求继续S3.2；启动pwd正确，已有S3.1学习确认的四份文档修改全部保留。
读取AGENTS、README、handoff、roadmap、S0.1模板、S3.1学习包/源代码；此前无S3.2包。
新增[学习包](vla/s3_2_multistep_rollout.md)、[脚本](../scripts/s3_2_multistep_rollout.py)、
[协议测试](../tests/test_s3_2_multistep_rollout.py)。仅本任务，不开始S3.3。

执行前核对解释器repo `.venv/bin/python`、prefix/base_prefix，Python3.12.3 / NumPy2.5.4，
实际NumPy路径 `/home/lucas/projects/embodied-ai-playground/.venv/lib/python3.12/site-packages/numpy/__init__.py`。
CPU单BLAS线程，无训练、新依赖、安装、dataset/weights下载或旧repo访问，requirements不变。
冻结 `tmp/s3_1_final` 的model.npz、episodes.npz、contract.json，记录各SHA256且结束前重核对。
拒绝未知schema、action/time/dynamics语义不符、坏shape/dtype/clock/split、非train-only统计。

10条原validation episode，原20actions/21states不变，按seed3200+episode_id生成延长部分固定动作；
同原synthetic dynamics延长到100步，dt.1s，同episode仅t0 reset，不拼接其他episode。
TF每步接收真实state，AR仅起点接收真值，全部预测器共用动作；无policy闭环。
保留float32 raw；MLP拼接输入float32沿用S3.1归一化，网络/状态输出float64。
首轮h1逐值一致性检查发现TF float32与AR float64基线加法精度差异（最大约1.2e-8m），
统一评价输入算术为float64、MLP接口显式float32、oracle遵循generator float32后通过；
没有放宽h1相等断言。原S3.1全部200条预测逐值精确复现。

Executed:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_2_multistep_rollout.py --output tmp/s3_2_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_2_multistep_rollout.py --bias-mm 2 --output tmp/s3_2_bias2_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_2_multistep_rollout.py --steps 40 --output tmp/s3_2_h40_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s3_2_multistep_rollout.py' -v
.venv/bin/python -m pip check
git diff --check
```

MLP h1/5/20/100 TF local RMSE .0983/.0924/.1042/.4649mm，
AR endpoint .0983/.3999/1.6074/15.5763mm；原S3.1 200条TF prefix RMSE .13786978mm。
默认MLP+1mm x偏置AR h100端点60.6893mm，bias2mm为128.2071mm，非严格翻倍。
不动/理想加法h100 AR端点175.8165/71.8391mm；oracle全部TF/AR零误差。
零动作identity解析探针TF L2 1/2mm，AR h100 L2 100/200mm，严格线性只属该探针。
H40时MLP端点3.9189mm，全部共有前缀不变；100步truth/MLP AR输入状态盒外比例各16.3%，
20步为2.0%/2.5%，长时误差混合递归传递与状态分布变化，盒内也不保证统计ID。
首轮100步batch10 TF+AR共200次批量推理p50/p95=2.595741/4.159114ms，3warmup/20samples；
含循环及finite/h1核验，不含真值生成、metrics、文件I/O、环境或planning，非机器人latency。

4/4独立协议测试通过：未来truth反事实不影响AR、oracle对齐零漂移、已知bias累积与独立reset、
延长保持source与共有前缀。独立inline核对artifact重载、各模型端点MSE重算、S3.1预测逐值重放、
source hash未变、bias不变量、H40/100所有数组共有前缀、6类CLI拒绝（已有目录/steps20/501/
biasNaN/11/缺source）、已有report不覆盖、action语义错配拒绝，全部PASS。
pip check无冲突；Markdown本地链接及git diff --check通过；产物ignored tmp/，未commit/push。
README/roadmap更新Engineering Complete，Run/Modify/Explain未勾选；只有用户反馈可确认掌握。
限制：仅toy固定动作预测，无学习uncertainty、无规划/控制、无机器人success/failure stage评价，
没有证明长时稳定/泛化，也未修复模型漂移。Current Task=S3.2；Next Task=S3.3仅计划。
用户先读Problem→Mathematics，再Run，先预测bias1→2mm后Modify，回答五问Explain；STOP。

## S3.2 learning confirmation — 2026-10-20

用户明确确认实验和预测均已完成，五项Explain准确区分TF每步真实状态输入与AR预测递归，
解释h1同初态同动作保证一致，TF h100仍只是使用真实s99的局部单步预测。
正确给出S3.1 200条转移每坐标MSE分母400、h1/h20端点分母20、h20 prefix分母400且不含t0。
准确写出误差递推恒等式和Jacobian近似，指出放大/压缩/抵消使误差不保证线性或单调。
明确bias1→2mm仅改变偏置预测器与解析探针，保持source/hash、truth、动作、普通MLP、基线与oracle；
零动作解析100步L2漂移100→200mm，而非线性MLP的RMSE不要求翻倍。
准确说明未来truth反事实、同动作及独立t0 reset审计；训练状态盒外16.3%说明坐标范围外推，
不能推出其余都是ID、盒外必失败、已校准不确定性或所有误差由OOD造成。
README与学习包Run/Modify/Explain已勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅修改四份状态文档，未重跑Python实验、安装或下载。
环境与实验结果沿用上述S3.2工程记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU），
不增加长时稳定、泛化、不确定性校准或真实机器人成功声明。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对仅四份文档修改。
Current Task=S3.2（已掌握）；Next Task=S3.3 — Model Uncertainty，仅计划。
等待用户选择，不自动开始ensemble或下一任务；未commit/push。

## S3.3 Engineering — 2026-10-20

用户明确请求推进S3.3；启动pwd正确，已有S3.2学习确认四份文档修改全部保留。
读取AGENTS、README、handoff、roadmap、S0.1模板、S3.1源码和S3.2接口/学习包；此前无S3.3包。
新增[学习包](vla/s3_3_model_uncertainty.md)、[脚本](../scripts/s3_3_model_uncertainty.py)、
[测试](../tests/test_s3_3_model_uncertainty.py)。仅S3.3，不开始S3.4或risk/planning。

执行前核对repo `.venv/bin/python`、prefix/base_prefix，Python3.12.3 / NumPy2.5.4，
实际NumPy路径 `/home/lucas/projects/embodied-ai-playground/.venv/lib/python3.12/site-packages/numpy/__init__.py`。
CPU单BLAS线程，无新依赖、安装、dataset/weights下载、旧repo访问或旧checkpoint修改，requirements不变。
自包含重建S3.1 deterministic generator/split；30train episodes/600transitions，10evaluation/200。
另独立seed330生成10名义calibration episodes/200transitions，只设置score q95阈值。
训练共享x/delta统计只fit原train600；K5个4→32→2 MLP共1130参数，各1500步full-batch Adam。
每成员按完整episode有放回采30次，seed3300+k；初始化seed3400+k，保持20转移episode边界。
raw float32，模型float64；动作仍world commanded XY displacement[m]，dt.1s，无success标签。

四个配对单步评价组各200条：ID、当前state x+.8m、action×3、same-input next-x+10mm隐藏机制变化。
OOD是反事实tuple而非连续trajectory；隐藏条件显式改变ground-truth机制，不改动作语义或预测。
D=sqrt(mean over K members and2coordinates of centered physical next-state prediction²)，ddof0，单位m。
flag=D>独立calibration q95；bad=ensemble mean L2 error>.005m是人工诊断标签，非机器人failure。
全部成员/均值物理误差、score分位数/相关性、逐样本及全分母confusion均保留；不调OOD阈值。

Executed:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_3_model_uncertainty.py --output tmp/s3_3_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_3_model_uncertainty.py --shift-mm 20 --output tmp/s3_3_shift20_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_3_model_uncertainty.py --members 2 --output tmp/s3_3_k2_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s3_*.py' -v
.venv/bin/python -m pip check
git diff --check
```

默认threshold .177804622mm；ID/state/action/hidden均值预测RMSE为.125052/2.946716/9.238062/7.068643mm，
score中位数.065647/1.251255/.651191/.065647mm；flagged1/200/168/1条，高误差0/49/110/200条。
TP/FP/FN/TN分别0/1/0/199、49/151/0/0、110/58/0/32、1/0/199/0。
各组score与error Pearson约.6451/.7765/.8012/.00365，仅描述相关性，不是校准。
复制第一个成员K次：score中位数0，隐藏机制200条高误差全部漏报。
shift20mm时所有权重/预测/score/threshold与其他条件不变，hidden RMSE14.139159mm，仍漏报199/200。
K2 threshold .114464mm、ID RMSE.125224mm；state/action高误差34/113条且FN1/6，hidden FN197/200。
K2前两个成员权重与bootstrap IDs等于K5前两个，不能从成员数推出单调提升或独立错误。
默认batch1 ensemble推理+mean/score p50/p95 .033752/.066264ms，20warmup/200samples，
不含训练、文件、真值、环境或planning。不是机器人latency或闭环成功证据。

最终9/9 Stage3协议测试通过（S3.2四项回归+S3.3五项）：population score分母/单位、共同偏置/
成员排列不变性、same-input target变化盲点、严格threshold混淆矩阵、NaN/rank/单成员拒绝。
独立inline核对checkpoint重载逐值复现全部成员预测和score、K2成员前缀、bootstrap只train、
train统计重算及validation+100m不影响train统计、calibration阈值重算、物理MSE与confusion分母200、
shift20除hidden target/error外所有数组逐值不变、state/action OOD语义；全部PASS。
6类CLI非法参数/已有目录拒绝（members1/11、epochs0、shiftNaN/51、已有output），report不覆盖。
pip check无冲突；Markdown本地链接及git diff --check通过；所有产物ignored tmp/，未commit/push。
README/roadmap更新Engineering Complete，学习三项未勾选，只有用户报告可确认Learning Mastered。
限制：deterministic toy单步分歧，不估计aleatoric噪声、不提供概率/coverage保证、不可靠检测任意OOD；
无policy/closed-loop success/failure-stage或规划实验。Current Task=S3.3；Next Task=S3.4仅计划。
用户读Problem→Mathematics→Run→先预测shift10→20mm后Modify→五问Explain；STOP。

## S3.3 learning confirmation — 2026-10-20

用户明确确认实验和预测均已完成，五项Explain准确覆盖episode bootstrap与不同初始化的多样性，
完整episode相关性、原train600拟合共享统计、calibration仅定阈值及held-out evaluation隔离。
正确写出K×N×2预测均值和population RMS分歧公式，解释2K分母、ddof0、单位m，
区分无需真值的score、单样本L2、逐坐标RMSE与ensemble均值standard error。
明确q95来自独立名义calibration200个score，flag为D>阈值、bad为真实L2>5mm诊断标签，
经验分位数不保证评价集5%误报率，也不提供机器人失败概率或安全风险。
准确指出shift10→20mm只改变隐藏机制target及误差相关诊断，保持权重/bootstrap/统计/
calibration/阈值、其他组结果与hidden输入/预测/score/flag；解释hidden RMSE7.069→14.139mm
仍漏报199/200，以及复制成员score约0仍可全错。
区分epistemic proxy、未估计的aleatoric噪声与条件转移机制变化；说明输入OOD不必然失败、
更多成员不保证准确/可信、低分歧不证明低误差或真实控制安全。
README与学习包Run/Modify/Explain已勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅修改四份状态文档，未重跑Python实验、安装或下载。
环境与工程结果沿用上述S3.3记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU），
不增加不确定性校准、任意OOD检测、机器人安全或闭环成功声明。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对仅四份文档修改。
Current Task=S3.3（已掌握）；Next Task=S3.4 — Latent State Representation，仅计划。
等待用户选择，不自动进入下一任务；未commit/push。

## S3.4 Engineering — 2026-10-20

用户明确请求推进S3.4；启动pwd正确，已有S3.3学习确认的四份文档修改全部保留。
读取AGENTS、README、handoff、roadmap、S0.1模板、S3.1源码与S3.3学习包；此前无S3.4包。
新增[学习包](vla/s3_4_latent_state.md)、[脚本](../scripts/s3_4_latent_state.py)、
[测试](../tests/test_s3_4_latent_state.py)。仅本任务，未开始S3.5或规划。

执行前确认repo `.venv/bin/python` / prefix/base_prefix，Python3.12.3 / NumPy2.5.4，实际模块路径
`/home/lucas/projects/embodied-ai-playground/.venv/lib/python3.12/site-packages/numpy/__init__.py`。
CPU单BLAS线程，无新依赖/安装/dataset/weights下载或旧repo访问，requirements不变。
复用S3.1 generator与episode split（seed31/32），不改旧checkpoint；新增6路已知线性投影观测，
profile s3_4_linear_sensor_latent/1.0。raw state/obs/action float32物理m，clock float64、dt.1s。
六路投影观测不是图像，encoder为train拟合的线性PCA，不声称神经autoencoder或视觉学习。

train30episodes的21观测各取一次，共630条fit观测mean/std及SVD；只取前d主方向，冻结E/D。
dynamics对600个train转移fit独立输入/delta_z统计，(d+2)→32 tanh→d residual MLP，
full-batch Adam .01/2000步，seed340；d2/d1分别226/161参数。
所有validation210观测/200转移未进统计/梯度，latent无量纲；每步命令仍world XY位移[m]。
物理评价将decoded六路观测经已知sensor伪逆恢复XY，是特权synthetic几何，不是未知sensor标定学习。
未来编码仅监督或诊断reference；AR20仅从E(o0)初始化，不看中途真值，不执行policy。

Executed final code:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_4_latent_state.py --output tmp/s3_4_final
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_4_latent_state.py --latent-dim 1 --output tmp/s3_4_d1_final
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s3_*.py' -v
.venv/bin/python -m pip check
git diff --check
```

首轮另运行tmp/s3_4_verified、tmp/s3_4_d1_verified；最终版清理未使用SVD调用，并将未来编码
重建诊断命名为reference而非floor，避免误称物理误差严格下界，数值模型结果相同。
d2保留观测方差≈100%，validation物理重建RMSE2.35952e-6mm（接近raw舍入），
单步decoded physical RMSE .116069mm、zero-action消融17.285216mm、latent persistence17.286918mm，
AR20 endpoint .758313mm。d2 train单步physical RMSE .085263mm。
d1保留55.674615%train标准化观测方差，validation物理重建121.208062mm、单步121.900999mm，
future encoding reference121.894283mm、AR20 endpoint154.780218mm。
d2/d1 raw latent prediction MSE1.09065e-6/7.22123e-5，对应latent persistence .0247726/.0247894；
不同表示loss不作物理精度排名，d1虽预测latent更好仍缺物理状态信息。
4tensor梯度有限差分最大绝对误差d2约1.985e-11/d1约4.324e-11。
d1在已知encoder物理null方向构造相距.1m、同z（误差<1e-12）两state，同[.03,.02]m动作
后的真实next-z差≈.001666315，证明该压缩表示对这两个状态不满足确定性动力学充分性。
首轮d2 batch1 encode/predict/decode p50/p95 .009404/.010985ms，20warmup/200samples，
不含sensor生成、真值、物理评价、I/O或planning；非机器人端到端latency。

13/13 Stage3协议测试通过（此前9项+本轮4项）：d2 held-out重建、d1 aliasing、未来observation
反事实不影响AR且h1一致、无效/退化encoder输入拒绝。
独立inline核对raw/split/d2→d1数据不变量、train630 PCA重算、validation+100m不影响encoder、
train600 latent统计重算、checkpoint单步与完整AR逐值重载、物理MSE/样本数、共有PCA轴/奇异值、
5类CLI拒绝（d0/d3/epochs0/10001/已有output）及report不覆盖，全部PASS。
该inline首轮发现d1 batch200单步与batch10 AR首步仅2.776e-17无量纲舍入差，
随后用atol1e-14/rtol0核验这项数值等价；checkpoint同计算路径重载仍要求逐值相等。
pip check无冲突；Markdown本地链接及git diff --check通过；所有产物ignored tmp/，未commit/push。
README/roadmap更新Engineering Complete，学习三项未勾选，仅用户反馈可确认掌握。
限制：已知线性无噪声sensor、名义toy held-out，无视觉/OOD/未知sensor辨识或真实robot任务；
没有success/failure-stage或闭环控制证据，低latent loss不等于状态充分或部署能力。
Current Task=S3.4；Next Task=S3.5仅计划。用户读Problem→Mathematics→Run→先预测d2→d1后Modify→五问Explain；STOP。

## S3.4 learning confirmation — 2026-10-20

用户明确确认实验和预测均已完成，五项Explain准确覆盖E/D/F的shape、物理单位与无量纲latent，
train-only PCA/SVD线性编码解码与神经autoencoder区别、E/D先fit再冻结。
正确说明train30×21=630观测与30×20=600转移，terminal用于离线encoder拟合合法，
无第21个动作；validation210观测/200转移不参与训练/统计，推理不使用未来观测。
准确区分重建、latent预测、decoded physical error和future-encoding特权reference，
后者不能作为部署预测或物理误差严格下界；不能仅以raw latent MSE跨表示排名。
正确列出d2→d1的raw/split/clock/测量矩阵/观测统计/奇异值不变量，及PCA/MLP shape变化；
d1保留55.6746%标准化观测方差、物理预测RMSE约121.901mm，相同z/动作却不同next-z
说明表示动力学不充分，不仅是优化不足。
准确说明AR20仅E(o0)初始化，未来观测反事实隔离、h1/动作索引/episode边界核验，
线性无噪声已知sensor与位置Markov toy不证明视觉理解、真实系统Markov或机器人成功。
助手补充接口边界：D只恢复六路观测，已知sensor伪逆是独立物理评价步骤，不属于learned decoder。
README与学习包Run/Modify/Explain已勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅修改四份状态文档，未重跑Python实验、安装或下载。
环境与工程结果沿用上述S3.4记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU），
不增加视觉/OOD/真实sensor辨识或闭环成功证据。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对仅四份文档修改。
Current Task=S3.4（已掌握）；Next Task=S3.5 — Reward / Success / Contact Prediction，仅计划。
等待用户选择，不自动进入下一任务；未commit/push。

## S3.5 Engineering — 2026-10-20

用户明确请求推进S3.5学习；启动pwd正确，已有S3.4学习确认四份文档修改全部保留。
读取AGENTS、README、handoff、roadmap、S0.1模板、S3.1源码与S3.3多头相关接口；此前无S3.5包。
新增[学习包](vla/s3_5_outcome_heads.md)、[脚本](../scripts/s3_5_outcome_heads.py)、
[测试](../tests/test_s3_5_outcome_heads.py)。仅本任务，不开始S3.6/规划或MPC。

执行前核对repo `.venv/bin/python` / prefix/base_prefix，Python3.12.3 / NumPy2.5.4，实际模块路径
`/home/lucas/projects/embodied-ai-playground/.venv/lib/python3.12/site-packages/numpy/__init__.py`。
CPU单BLAS线程，无新依赖、安装、dataset/weights下载、旧repo访问或旧checkpoint修改；requirements不变。
新profile s3_5_wall_outcomes/1.0显式改变动力学：S3.1 attempted transition后next_x截断到.12m，
contact=attempted_x>.12m（主动墙约束，无force/真实contact solver）。
固定goal[.09,0]m/radius.045m，success_next是下一状态goal occupancy；failure_next是next_x<−.2m或|y|>.2m。
reward=(distance_before−distance_after)/.04m−.2contact−.5failure_next，无量纲。
四标签各有定义但不假定统计独立；episode_success=terminal occupancy且全episode无failure，仅结果不入head。
所有episode20转移/21状态、dt.1s，只有末步truncated，无事件termination；失败之后轨迹也全部保留。

新生成60个独立reset，四种手写采集行为random/goal/right-push/outward，不是本轮训练policy。
seed350/351，train40episodes/800转移、validation20/400；raw state/action/reward float32，事件bool，clock float64。
模型输入仅当前XY+候选world displacement[m]，不含next_state/contact/result；input/reward统计只fit train800。
4→32 tanh→4共292参数，reward linear + success/contact/failure三个sigmoid logits，float64 CPU。
loss=标准化reward MSE+三个binary BCE均值，显式梯度/稳定logaddexp/sigmoid，Adam .01/2500步，seed352。
默认threshold.5与学习对照.2固定指定，不从验证/OOD选优；有界分数不等于校准机器人风险。

Executed:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_5_outcome_heads.py --output tmp/s3_5_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_5_outcome_heads.py --threshold .2 --output tmp/s3_5_t02_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s3_*.py' -v
.venv/bin/python -m pip check
git diff --check
```

train/validation reward RMSE .078729/.152957，validation常量train-mean基线.735068、zero-action消融.481000。
默认validation success/contact/failure正样本72/131/138，各组N400，TP/FP/FN/TN分别
46/10/26/318、129/0/2/269、132/1/6/261；Brier .063394/.003827/.018685。
动作×3反事实重新执行墙toy/标注，非连续OOD轨迹；reward RMSE1.090921，常量1.837897，zero-action1.537491。
OOD三head正样本73/155/156，TP/FP/FN/TN 44/31/29/296、150/1/5/244、112/2/44/242。
threshold .2时validation混淆矩阵60/10/12/318、130/0/1/269、132/5/6/257；OOD failure FN44→33、FP2→3。
阈值对照所有NPZ数据、权重、reward预测、probability逐值相同，Brier/BCE不变，仅decision/相关计数改变。
首轮batch1全heads推理p50/p95 .008156/.008616ms，20warmup/200samples，含标准化/MLP/sigmoid，
不含sensor/I/O/环境执行/标签计算/planning；不是机器人闭环latency。
4tensor有限差分最大绝对误差1.126e-11；另对四个输出bias逐个核验梯度。

18/18 Stage3测试通过（此前13项+本轮5项）：标签非等价组合、执行后时序、墙截断、threshold方向性、
四head梯度与±1000 logits稳定性。独立inline核对raw转移逐值、四标签重算、episode_success及boundary，
episode split、train-only统计/validation+100m隔离、checkpoint预测逐值复现、OOD重标注、reward与分类指标，
所有confusion分母400、threshold全部NPZ不变量和FP非减/FN非增、6类CLI非法/已有目录拒绝及不覆盖，全部PASS。
pip check无冲突；Markdown本地链接及git diff --check通过，产物ignored tmp/，未commit/push。
README/roadmap更新Engineering Complete，学习三项未勾选，只有用户反馈可确认掌握。
限制：toy一步事件不等于最终任务success/failure；contact是几何截断无真实force，failure score无校准保证。
未训练head-driven policy、未集成latent/动力学或执行规划/机器人闭环；collection结果不是head控制成功证据。
Current Task=S3.5；Next Task=S3.6仅计划。用户读Problem→Mathematics→Run→先预测threshold.5→.2后Modify→五问Explain；STOP。

## S3.5 learning confirmation — 2026-10-20

用户明确确认实验和预测均已完成，五项Explain准确区分四个一步输出、episode_success与
terminated/truncated：事件不终止本toy episode，20步time limit才截断。
正确写出attempted nominal transition、右墙clip/contact、goal occupancy、边界failure与
progress/contact/failure reward公式，说明正reward不等于success、contact不等于failure，
独立语义定义不等于统计独立。
准确给出4→32→4/292参数、输入当前state/action [m]、reward无量纲与三个sigmoid分数、
标准化reward MSE+三head平均BCE；事件非互斥，因此不用softmax，next_state只作标签来源。
明确threshold.5→.2时数据/权重/统计/reward/分数/Brier/BCE不变，TP/FP非减、FN/TN非增，
复述validation success FN26→12、contact FN2→1、failure FN6不变且FP1→5。
正确说明failure分数只预测一步边界事件，未校准为机器人未来失败风险。
准确比较reward train/validation/OOD误差与failure FN6/138→44/156，区分保留原标签的
zero-action输入消融和实际执行零动作；离线loss/contact accuracy/手写采集结果不构成
head-driven policy/planner或真实机器人闭环成功、安全证据。
README与学习包Run/Modify/Explain已勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅修改四份状态文档，未重跑Python实验、安装或下载。
环境与工程结果沿用上述S3.5记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU），
不增加真实contact物理、风险校准或闭环成功声明。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对仅四份文档修改。
Current Task=S3.5（已掌握）；Next Task=S3.6 — Model-based Planning，仅计划。
等待用户选择，不自动进入下一任务；未commit/push。

## S3.6 Engineering — 2026-10-20

用户明确请求推进S3.6学习；启动pwd正确，已有S3.5学习确认四份文档修改全部保留。
读取AGENTS、README、handoff、roadmap、S0.1模板、S3.1/S3.3训练接口与S3.5环境/数据代码；此前无S3.6包。
新增[学习包](vla/s3_6_model_based_planning.md)、[脚本](../scripts/s3_6_model_based_planning.py)、
[测试](../tests/test_s3_6_model_based_planning.py)。仅本任务，未实现S3.7 feedback/replanning。

执行前核对repo `.venv/bin/python` / prefix/base_prefix，Python3.12.3 / NumPy2.5.4，实际模块路径
`/home/lucas/projects/embodied-ai-playground/.venv/lib/python3.12/site-packages/numpy/__init__.py`。
CPU单BLAS线程，无新依赖/安装/dataset/weights下载/旧repo访问或旧checkpoint修改；requirements不变。
重新生成S3.5 wall source seed350/split351，train40/800、held-out20/400；匹配墙数据重新训练226参数
residual dynamics，seed360、Adam.01/2000步，train-only input/delta统计，未混用S3.1无墙checkpoint。
数据raw float32、模型float64；action仍world commanded displacement[m]，dt.1s。

新profile s3_6_wall_planning/1.0，20个held-out t0初态全部保留。默认N128/H10，候选index0零动作，
其余按episode/candidate-specific seed生成两段随机命令与active duration，余下是真实零命令非padding。
N32是N128严格前缀，生成器不看goal/真实未来；每轴命令≤.06m。
learned rollout→known dimensionless task cost→argmin，仅t0规划，所选10步整段open-loop执行。
cost=terminal距离²/.1m²+.05mean路径距离²/.1m²+5sum未来边界违规+.01mean命令²/.06m²。
不接S3.5 learned reward/risk heads或S3.4 latent；选择先于该reset任何真实候选future评价。
同初态/候选比较learned、ideal wall-aware addition、zero、fixed random index1、特权plant cost oracle。
实际plant nominal为S3.5 wall step；weak_gain在plant内部将命令×.65后step，planner不知gain且选同一序列。
成功=最终goal occupancy且10个future states无边界违规；失败分boundary_violation/goal_not_reached。
只有toy数值plant，无真实机器人或MPC闭环success声明。oracle仅同bank最小cost，不是全局或最大成功率上界。

Executed final code:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_6_model_based_planning.py --output tmp/s3_6_final
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_6_model_based_planning.py --candidates 32 --output tmp/s3_6_n32_final
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s3_*.py' -v
.venv/bin/python -m pip check
git diff --check
```

首版另运行tmp/s3_6_verified/tmp/s3_6_n32_verified；最终版增加candidate_success_count与coverage/selection
诊断，模型/执行结果相同。model.npz SHA256记录report且评价末尾核验未变。
单步held-out RMSE .373134mm。N128 nominal learned/ideal/zero/random/oracle成功19/8/0/0/19（各20次），
weak_gain为6/2/0/0/15。learned nominal失败0 boundary/1 goal，weak_gain 0 boundary/14 goal。
learned nominal/weak_gain平均actual cost .128948/.490862，bank regret .002018/.242964，
selected终点预测L2均值3.768/65.097mm；预测成功均19，实际虚报0/13。
nominal1个bank无成功candidate，learned选错可成功bank为0；weak_gain5个bank无成功candidate，
另9个bank有成功candidate但learned选择失败。所有20初态/每组完整分母保留，不筛失败。
N32 nominal learned/oracle13/13，7个bank无成功candidate；weak_gain learned/oracle6/8，12个bank无成功，
learned另2个bank有成功candidate却选失败。zero/random实际轨迹与cost不变，但bank oracle变动会改变其regret。
最终N128 learned rollout/score/select p50/p95 .683442/1.592837ms，generation1.507280/3.861850ms；
N32对应.223997/.678625ms、.389721/1.073543ms，各20reset samples/warmup0。
两阶段p95不可直接相加作端到端p95；不含训练、真实plant、文件I/O或robot执行，非deadline保证。

23/23 Stage3测试通过（此前18项+本轮5项）：嵌套候选/限幅/零动作、递归与reset隔离、
patch真实plant拒绝以验证选择不访问plant、argmin与失败分期、nonfinite拒绝。
独立inline核对N128/N32模型与hash一致、source/split/初态一致、候选/ideal/真实plant前缀逐值相同、
learned预测/score前缀atol1e-12数值等价、嵌套min predicted cost单调、train统计重算、checkpoint完整rollout逐值重放；
每attempt所选序列重新在独立plant执行逐值重现，argmin/oracle/cost/coverage/regret/成功重算，
各summary success+failure=20、跨plant learned choice一致、zero/random实际cost不变；全部PASS。
5类CLI非法/已有目录拒绝（N1/N513/epochs0/10001/已有output），report不覆盖。
pip check无冲突；Markdown本地链接及git diff --check通过；产物ignored tmp/，未commit/push。
README/roadmap更新Engineering Complete，Learning三项未勾选，仅用户反馈可确认掌握。
限制：固定goal/wall、20reset/单seed、有限candidate coverage；低离线loss不保证执行器变化下的规划表现，
无视觉语言policy/真实接触力/风险校准或机器人闭环证据，也未用反馈修复weak_gain失败。
Current Task=S3.6；Next Task=S3.7仅计划。用户读Problem→Mathematics→Run→先预测N128→32后Modify→五问Explain；STOP。

## S3.6 learning confirmation — 2026-10-20

用户明确确认实验和预测均已完成，五项Explain准确区分candidate generation、learned rollout、
known cost、argmin与actual plant；oracle需要所有候选真实未来，只能作特权参照。
正确给出world XY state/command位移[m]、dt.1s、source/train/bank/rollout/cost shape，
说明S3.1无墙模型不匹配S3.5墙裁剪动力学，应使用对应环境数据拟合。
准确解释无量纲cost的终点/路径/违规/能量项，区分cost最优与二值success、candidate coverage，
以及只相对当前bank的actual regret，非全局最优差距。
明确N128→32保持数据/统计/权重/初态与前32候选动作及预测/实际轨迹，
嵌套bank只保证min predicted cost(128)≤min predicted cost(32)，不保证真实success单调。
正确指出zero/random实际动作/cost不变，regret可因oracle参照变化而变；复述nominal19→13、weak_gain6→6。
准确区分nominal19/20的toy效果、未知.65gain下6/20的执行机制失配与teacher-forced单步RMSE.373mm。
说明episode split/train-only统计/先选择后评价的泄漏审计，利用coverage及真实cost/终点误差排查候选不足和选错。
全程t0单次规划、整段10步执行，不是MPC或物理机器人闭环证明。
README与学习包Run/Modify/Explain已勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅修改四份状态文档，未重跑Python实验、安装或下载。
环境与工程结果沿用上述S3.6记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU），
不增加未知机制鲁棒性、真实机器人、MPC或风险校准声明。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对仅四份文档修改。
Current Task=S3.6（已掌握）；Next Task=S3.7 — MPC-style Replanning，仅计划。
等待用户选择，不自动进入下一任务；未commit/push。

## S3.7 Engineering — 2026-10-20

用户明确请求推进S3.7；启动pwd正确，已有S3.6学习确认四份文档修改全部保留。
读取AGENTS、README、handoff、roadmap、S3.6源码/学习包、S3.5 plant接口及既有S0.1模板约定；此前无S3.7包。
新增[学习包](vla/s3_7_mpc_replanning.md)、[脚本](../scripts/s3_7_mpc_replanning.py)、
[测试](../tests/test_s3_7_mpc_replanning.py)。仅本任务，不开始S3.8或Stage4。

执行前核对repo `.venv/bin/python` / prefix/base_prefix，Python3.12.3 / NumPy2.5.4，实际模块路径
`/home/lucas/projects/embodied-ai-playground/.venv/lib/python3.12/site-packages/numpy/__init__.py`。
CPU单BLAS线程，无训练/新依赖/安装/dataset/weights下载/旧repo访问，requirements不变。
只读冻结 `tmp/s3_6_final` model/contract/planning/report；loader检查schema/plant profile、model shape/
finite/positive scale、source wall transition对齐、完整episode split、held-out初态及train-only统计，
模型SHA256匹配source report，全部四文件hash前后不变；不隐式重训或估计gain。

新profile s3_7_wall_feedback/1.0：相同20个held-out starts、N128、固定T10/dt.1s，raw world XY
state/command float32[m]，模型float64。三方法open_loop(E10)、replan_no_feedback(E CLI)、mpc_feedback(E CLI)。
每次remaining horizon=10-t，执行L=min(E,10-t)，E4在t0/4/8规划并执行4/4/2，E1十次、E10一次。
feedback仅在规划边界接收实际当前XY；无反馈用旧计划预测prefix终点，模型/统计都不更新。
S3.6 bank(episode_id+1000*t,N,remaining)，t0相同；后续last candidate替换为未执行旧后缀，总数仍N，
方法间旧后缀可能因历史反馈不同而不同。所有动作均真实执行，不早停、不拼接reset。
known cost复用S3.6但平均项按remaining length；非严格Bellman一致或稳定性保证。
nominal、weak_gain .65（plant内部）、kick（transition4→5后y+.08m）三条件，无未来预告。

Executed:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_7_mpc_replanning.py --output tmp/s3_7_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_7_mpc_replanning.py --execute 4 --output tmp/s3_7_e4_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_7_mpc_replanning.py --execute 10 --output tmp/s3_7_e10_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s3_*.py' -v
.venv/bin/python -m pip check
git diff --check
```

原S3.6离线RMSE .373134mm与全部source hash不变。E1 nominal open/no-feedback/feedback成功19/20/20，
weak_gain6/3/19，kick2/0/20（各20attempts）；E4 nominal19/20/20、weak6/3/18、kick2/0/20。
本轮各组所有失败均goal_not_reached，boundary_violation0，未删除失败；无早停，恰好10actions/11states。
E10三方法完整轨迹逐值相同，success nominal19、weak6、kick2；nominal/weak open-loop逐值复现S3.6。
E1 feedback平均final distance nominal/weak/kick 2.910/11.778/2.846mm，E4为4.826/17.890/11.532mm。
feedback calls/episode E1为10、E4为3，三条件共600/180次调用，计算更多不是同预算open-loop对照。
no-feedback实际动作跨三条件逐值一致，证明重新搜索没有引入实际state；feedback则按真实偏差修正起点。

E1 feedback每次generation+selection p50/p95 ms：nominal2.072/5.132、weak2.356/4.032、kick2.321/4.098；
每episode总量21.604/39.613、23.404/31.837、23.841/34.048ms。
E4每episode总量7.626/11.655、6.932/9.323、7.381/10.478ms，实测两段相加后再统计，warmup0。
按每次100ms×实际prefix长度对照的compute budget overrun均0；plant时间单列，不含sensor/I/O/调度，
模拟plant不随compute等待继续演化，不能称真实机器人deadline或时延鲁棒性证明。

27/27 Stage3测试通过（此前23项+本轮4项）：feedback起点/E4短末前缀、无反馈动作不受plant变化影响、
E10等价退化、warm仅保留未执行后缀。独立inline逐次重建三run全部bank/cost/argmin/所选动作并核对
observed-state vs model-belief，逐plant step重放全部raw轨迹、outcome/actual cost与20分母；
核对source freeze、E10三方法逐值相同、S3.6 open-loop精确复现、不同E open-loop不变、
无反馈动作跨plant不变，全部PASS。6类CLI非法/已有目录拒绝（E0/E11/N1/N513/缺source/已有output），report不覆盖。
pip check无冲突；Markdown本地链接及git diff --check通过；生成数据/记录全部ignored tmp/，未commit/push。
README/roadmap更新Engineering Complete，Learning三项未勾选，只有用户反馈可确认掌握。
限制：理想无噪声XY观测、固定goal/wall/单seed20starts；MPC-style finite shrinking horizon，
无模型适应/未知gain识别/视觉policy/真实contact或机器人实验，不提供一般稳定性或风险保证。
Current Task=S3.7；Next Task=S3.8仅计划。用户读Problem→Mathematics→Run→先预测E1→4后Modify→五问Explain；STOP。

## S3.7 learning confirmation — 2026-10-20

用户明确确认实验和预测均已完成，五项Explain准确区分open_loop的t0真值、
replan_no_feedback的模型预测prefix终点与mpc_feedback的已执行真实XY；重复搜索不等于观测反馈。
正确写出T10/R=10-t/L=min(E,R)，列出E1 t0…9、E4 t0/4/8与4/4/2、E10单次规划，
说明warm仅取chosen[L:]替换最后candidate、N不变，bank(N,R,2)/prediction(N,R+1,2)，单位m与dt.1s不变。
准确解释冻结权重和normalizer仍可通过实际起点重置纠正gain引起的累计偏差，但没有模型适应、
gain估计或稳定性/鲁棒性保证。
明确E1→4保持checkpoint/统计/初态/T/plant/open-loop轨迹，改变观测频率、计算量及后续warm/candidate/选择，
复述weak_gain feedback19→18/20而nominal/kick20/20，actual success无单调保证。
准确给出source hash、train/validation隔离、planning state来源、bank/cost/argmin/plant重放、
E10退化逐值一致等审计方法，解释no-feedback weak_gain3/20揭示搜索不能替代反馈。
所有20次/10步固定、失败全保留；latency不含真实sensor/通信/调度和compute期间plant演化，
不能当作物理机器人实时闭环证明。
README与学习包Run/Modify/Explain已勾选，roadmap同步Learning Mastered。
启动pwd正确，git status --short为空；本轮仅修改四份状态文档，未重跑Python实验、安装或下载。
环境与工程结果沿用上述S3.7记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU），
不增加模型适应、一般闭环稳定性、真实机器人或deadline保证。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对仅四份文档修改。
Current Task=S3.7（已掌握）；Next Task=S3.8 — Video / Generative World Model Concepts，仅计划。
等待用户选择，不自动进入下一任务；未commit/push。

## S3.8 Engineering — 2026-10-20

用户明确请求推进S3.8；启动pwd正确，已有S3.7学习确认四份文档修改全部保留。
读取AGENTS、README、handoff、roadmap和S0.1模板；此前无S3.8包。本轮仅S3.8，不开始Stage4。
新增[学习包](vla/s3_8_video_world_model_concepts.md)、[sources](vla/s3_8_sources.json)、
[脚本](../scripts/s3_8_generative_world_concepts.py)、[测试](../tests/test_s3_8_generative_world_concepts.py)。

官方研究：DreamerV3 arXiv2301.04104v2 (2024-04-17) Figure3/World model learning/Equation1，
GAIA-1 arXiv2309.17080v1 (2023-09-29) Figure2/Model/Video Decoder及sampling，
Genie2官方2024-12-04介绍的action/counterfactual/latent diffusion架构；不混用GAIA2/Genie1或Genie3。
`git ls-remote https://github.com/danijar/dreamerv3.git HEAD`返回
`e01491fad6434b2245a3b8ca201dd7faedcc458c`，静态读该revision rssm.py observe/_observe/imagine和
agent.py loss/model/imagination相关片段，未声称与历史论文逐行同版本。
Nature全文访问被重定向到不可访问认证页，实际Dreamer依据使用可访问固定arXiv v2与固定官方源码；
未执行上游、安装JAX、下载仓库/视频样本/dataset或weights。所有URL、日期、访问限制写入manifest。

环境执行前核对repo `.venv/bin/python` / prefix/base_prefix，Python3.12.3 / NumPy2.5.4 / Pillow12.3.0；
实际模块 `.venv/lib/python3.12/site-packages/numpy/__init__.py` 与 `PIL/__init__.py`，无新依赖/安装，requirements不变。
本地profile s3_8_branching_video/1.0是独立解析toy，无墙/no contact任务，不沿用S3.5标签。
20reset、左右两action、两固定隐藏mode，80真轨迹；8转移/9观测、dt.1s，动作±.015m x命令，
隐藏b±1在整段固定，y每步漂移b*.0075m；初始画面完全相同。先验已知、未训练任何model。
初态/action float32[m]，解析latent/state float64[m]，RGB32×32 uint8；已知位置与renderer均特权脚本逻辑，
无learned视觉encoder/decoder、latent dynamics或denoiser。像素mean使用两真分支，是明确特权参考。

Executed final code:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_8_generative_world_concepts.py --output tmp/s3_8_final
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_8_generative_world_concepts.py --samples 1 --output tmp/s3_8_k1_final
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s3_*.py' -v
.venv/bin/python -m pip check
git diff --check
```

初版tmp/s3_8_verified/tmp/s3_8_k1_verified发现解析加法沿用float32，最终版显式float64计算以匹配契约；
报告/图片使用最终目录。K16 persistent/frame-resampled all-sample path MSE均.001434375m²，
前者transition violation0，后者44.5703125%、整段invalid99.0625%；exact support coverage100%/7.5%。
action-ignored coverage0、violation100%；mean-state path MSE.0007171875m²却violation100%。
mean pixels/mean-state render/copy initial frame pixel MSE .001211227/.003733637/.005030121，
末帧物体连通数真值1/mean pixels2；同初态左右action终点相距≈.24m。
K1 persistent coverage50%、best-of-K .001434375m²；K16 coverage100%、best-of-K0，
但all-sample MSE和sample0不变。best-of-K使用真未来挑整段sample，不等于单次部署预测/概率校准。
K1正分支采样比例30%、K16为50.625%，是有限采样而非prior更改；hidden mode不同于ensemble uncertainty。

31/31 Stage3测试通过（此前27项+本轮4项）：相同context动作反事实、持久/逐帧joint、K前缀/
覆盖vs单次误差、均值像素ghost。独立inline核对所有dtype/shape、K真值/动作/图像不变量、样本前缀、
renderer与均值重载、全部物理/分布metric、mean MSE解析值、左右共用随机mode、GIF9frames、PNG1044×624，
source manifest读取与3类CLI拒绝（K0/K65/已有output）及report不覆盖，全部PASS。
已用view_image检查最终comparison.png，GIF可解码；图像是脚本渲染而非上游模型结果。
pip check无冲突；Markdown本地链接及git diff --check通过，产物ignored tmp/，未commit/push。
无learned model，因此不报告虚构train loss、model inference latency或robot success；
未证明任何官方模型有本toy失败率，不做开放世界覆盖/真实物理准确或闭环控制声明。
README/roadmap同步S3.8 Engineering Complete；Stage3工程包全部完成，S3.8 Learning三项未勾选。
Current Task=S3.8；Next Task=S4.1 — VLA Proposal仅计划，不自动开始Stage4。
用户读固定架构表/公式→Run→先预测K16→1后Modify→五问Explain；STOP。

## S3.8 learning confirmation — 2026-10-20

用户明确确认实验和预测均已完成，五项Explain准确区分DreamerV3 latent posterior/prior与
actor/critic imagination、GAIA1 token dynamics/video diffusion decoder、Genie2动作条件latent diffusion；
generative不等于diffusion，时间AR与去噪可组合，latent控制不必先渲染视频。
正确给出初态/动作/真分支/sample/RGB shape、dtype、world XY单位与dt，解释整段固定hidden mode
不等于每帧iid噪声或ensemble disagreement。
准确说明非法mean-state的MSE低于合法采样、mean pixels产生ghost物体、render(mean state)不同，
以及frame-resampled边缘正确却破坏joint，复述44.5703%转移/99.0625%整段违例。
明确K16→1保持raw/规则/mean reference/sample0，persistent coverage100%→50%、best-of-K0→.001434375m²，
all-sample MSE不变；best-of-K使用完整真未来事后选样，不代表单次部署或机器人success。
正确说明prior imagination只能用历史/动作/预测latent/独立随机性，真实未来只能作监督/posterior或事后评价；
解析toy与官方演示不证明真实物理、校准、OOD、闭环或实时能力，需独立动作/物体/物理/多步/任务/latency评价。
README与学习包Run/Modify/Explain已勾选，roadmap同步；Stage3全部工程与学习完成。
启动pwd正确，git status --short为空；本轮仅修改四份状态文档，未重跑Python实验、安装或下载。
环境与工程证据沿用上述S3.8记录（repo .venv Python3.12.3 / NumPy2.5.4 / Pillow12.3.0 CPU），
不增加大型视频模型训练/复现或真实机器人能力声明。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对仅四份文档修改。
Current Task=S3.8（已掌握）；Next Task=S4.1 — VLA Proposal，仅计划。
等待用户选择，不自动开始Stage4；未commit/push。

## S4.1 Engineering — 2026-10-20

用户明确请求继续S4.1；启动pwd正确、git status --short为空。读取AGENTS/README/roadmap、
交接当前及历史记录、S0.1模板和S1.7当前来源包。仅推进S4.1，未实施S4.2。
[学习包](vla/s4_1_vla_proposal.md)、[script](../scripts/s4_1_vla_proposal.py)、
[tests](../tests/test_s4_1_vla_proposal.py)。

环境核对`.venv/bin/python` Python3.12.3、NumPy2.5.4，实际模块路径
`.venv/lib/python3.12/site-packages/numpy/__init__.py`；无安装/下载。
本地没有历史S1.5 weights.npz；先向用户说明，再显式以原seed7/2400steps重建CPU toy，
不声称新weights文件与历史Python3.14 source逐bit一致。proposal加载后weights readonly，
五份source文件SHA256前后一致；不训练，不重fit固定尺度或tokenizer。

Executed:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s1_5_tiny_transformer_vla.py --output tmp/s4_1_source_s1_5 > tmp/s4_1_source_stdout.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_1_vla_proposal.py --output tmp/s4_1_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_1_vla_proposal.py --candidates 1 --output tmp/s4_1_k1_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s4_1*.py'
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pip check
git diff --check
```

K5/K1输出(16,5,3,2)/(16,1,3,2)float32 world XY commanded displacement[m]，H3/dt.1s/每分量±.06m。
16known attempts生成提案、2unsupported attempts保留输入拒绝，无伪造chunk。
candidate0为原learned chunk，其余是固定±x/±y1cm手工偏置后clip；不是learned采样。
旧8validation布局重复使用，没有独立test；独立candidate0 E1诊断两次均0/16成功，
16条失败stage全final_goal_error_after_budget。无WM评分、排序、接受/拒绝坏动作或fallback。
K5 proposal median/P95 .134/.231ms，K1 .139/.190ms，各16次、5warmup排除；
计时含推理+bank+contract，不含render/IO/真实sensor/servo，不能作实时或稳定速度排名。

3项新增测试覆盖语义/embodiment/unit/frame/dt错配、坏payload、候选前缀/clip来源；
全34项测试通过。独立inline核对跨K source hash/offline/input/candidate0/闭环commands与trace完全一致，
并核对16条失败stage；修改文档本地链接、pip check与git diff --check通过。
所有产物ignored tmp，无新依赖、旧repo修改或commit/push。
README/roadmap同步Engineering Complete，Run/Modify/Explain全部未勾选。
Current Task=S4.1；Next Task=S4.2仅计划。
用户先读Problem→Mathematics，Run→先预测K5→1后Modify→五问Explain；STOP等待学习确认及选择。

## S4.1 learning confirmation — 2026-10-20

用户明确确认实验和预测均已完成；五项Explain准确覆盖输入/预处理/候选shape、dtype、world XY单位、
slot时序及不读未来RGB/真实goal/success；冻结candidate0与手工bias候选来源，不将确定性重复推理
或偏置称为learned多模态采样；K5→1模型/hash/tokenizer/固定尺度/输入/candidate0/offline/拒绝/
闭环commands与success不变量及H/K/E/dt区别，latency不保证单调。
正确区分声明检查与能力/安全、producer真实性，以及S3墙plant与ideal point的embodiment不匹配；
准确解释16/18输入提案与0/16闭环分母、8布局×2指令/8步、final_goal_error_after_budget，
输入拒绝不等于风险拦截，offline MSE、本地latency、34项测试均不能证明部署成功。
README、学习包Run/Modify/Explain已勾选，roadmap同步Learning Mastered；策略失败结论保留。
启动pwd正确、git status --short为空；本轮仅改四份状态文档，未重新运行实验、安装或下载。
环境和工程证据沿用S4.1记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU）。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对四份文档修改。
Current Task=S4.1（已掌握）；Next Task=S4.2 — World-model Verification，仅计划。
等待用户选择，不自动实施S4.2；未commit/push。

## S4.2 Engineering — 2026-10-20

用户明确请求继续S4.2。启动pwd正确；已有README/handoff/roadmap/S4.1 package四份学习确认修改，
全部保留并叠加当前任务文档。读取AGENTS/README/handoff/roadmap、S0.1模板、S4.1 package/script及
S3.1动力学源码。S4.2此前无学习包；本轮只实现本任务，未进入S4.3。
[学习包](vla/s4_2_world_model_verification.md)、[script](../scripts/s4_2_world_model_verification.py)、
[tests](../tests/test_s4_2_world_model_verification.py)。

环境核对repo `.venv/bin/python` Python3.12.3、NumPy2.5.4，实际NumPy模块在
`.venv/lib/python3.12/site-packages/numpy/__init__.py`，CPU；无新依赖、安装或下载。
先向用户解释不能把S3 gain/wall模型直接接ideal point。新增匹配S4.1 plant的10系数affine residual，
不是S3模型复用/大型world model训练：40独立episode、每20transitions；seed421数据/422整episode split，
30train600转移fit input mean/std与least squares，10validation200转移独立评价。
目标delta保留物理m，不fit target scaler。raw NPZ存完整episodes/split/timestamp，输出ignored tmp。
拟合后save/reload逐值核对、readonly冻结；候选AR只读initial+actions+model，无真实goal/语言/图像/未来/gain输入。

默认读取`tmp/s4_1_verified`的三份artifact，严格contract/schema/shape/dtype/finite/tick0与分母检查；
源文件SHA256前后一致。16supported×K5=80条独立reset rollout/条件，H3/4states/dt.1s；
2次unknown输入拒绝原样保留，不伪造未来，也不称为风险拦截。先生成AR预测再生成plant truth；
名义gain1保持ideal point；gain.65/.4明确是仅评价用的执行机制变化，模型不知情、不改变command语义。
另做TF真实pre-action state诊断与zero-action消融；不选候选、不做risk/success评分/拒绝或反馈replan。

Executed:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_2_world_model_verification.py --output tmp/s4_2_verified
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s4_2*.py'
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_2_world_model_verification.py --plant-gain .4 --output tmp/s4_2_gain04_verified > tmp/s4_2_gain04_stdout.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_2_world_model_verification.py --proposals tmp/s4_1_k1_verified --output tmp/s4_2_k1_verified > tmp/s4_2_k1_stdout.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pip check
git diff --check
```

默认validation单步RMSE6.823e-9m，名义候选AR RMSE2.409e-9m；zero-action消融23.984mm。
模型类恰含线性无噪声真plant，近零误差只反映简单可识别系统，非真实物理/泛化证据。
Gain.65 AR三个horizon RMSE2.498/6.443/12.792mm，全未来8.394mm；TF全转移4.813mm。
Gain.4 AR三个horizon4.283/11.046/21.930mm，全未来14.391mm；TF8.251mm。
K5每condition误差scalars480，排除共享initial；80候选来自旧8布局×2文本与相关bias，不是独立80场景。
默认AR3 batch80 latency median/P95 .0258/.0343ms，30次/5warmup；仅normalize/matmul/state update，
排除proposal/plant/metrics/IO/sensor/servo，无整机实时结论。无新controller，closed_loop_success=null；
S4.1原candidate0闭环0/16能力结论保留，不以预测准确声称控制改进。

5项新增测试通过，全39项测试通过：train-only fit隔离、独立reset/时间对齐、gain/未来不进入AR、
model语义/plant/统计错配拒绝、proposal语义先于payload加载拒绝。
独立inline解析核对gain.65/.4的AR误差≈(1−g)cumsum(actions)、TF误差≈(1−g)actions；
跨gain model/source/训练与AR/名义真值不变，K1候选0轨迹保留，dtype/time/train统计/source hash及NPZ round-trip通过。
额外拒绝已有output（report不被覆盖）、NaN gain、缺失source，均不创建无效output。
pip check、本地Markdown links、git diff --check通过；无旧repo修改、commit/push。
README/roadmap更新S4.2 Engineering Complete；学习Run/Modify/Explain全部未勾选。
Current Task=S4.2；Next Task=S4.3 — Risk / Success Scoring仅计划。
用户读Problem→Mathematics→Run→先预测gain.65→.4后Modify→五问Explain；STOP等待用户确认及选择。

## S4.2 learner review — 2026-10-20

用户明确报告实验/预测完成。Explain第3–5项正确覆盖AR/TF因果隔离、独立reset、gain误差公式与
12/7倍率、动作相消、18/80/480/160分母、近零预测/latency/测试的能力证据限制。
Run/Modify在README与学习包勾选；Explain未勾选，待用户补充第1–2项的实现精度：
raw truth=(16,5,4,2)float32，AR同shape为float64，TF=(16,5,3,2)float64且无初态；
原policy单例(3,2)、手工bank单例(5,3,2)、汇总16例(16,5,3,2)。
phi=[normalized current XY/command XY,1]，delta=phi@coefficients；系数(5,2)共10个，
input_mean/std各(4,)仅train拟合，30train600/10validation200，目标保留原delta[m]。
用户2×4矩阵+bias是等价数学写法，但尚未明确本实现归一化/拼接参数化。
保留本轮启动已有全部修改和未跟踪S4.2文件，仅调整四份状态文档；未重跑实验或训练。
检查解释器/NumPy路径仍为repo .venv Python3.12.3 / NumPy2.5.4；只读加载
`tmp/s4_2_verified/verification.npz`与`model.npz`，逐项打印shape/dtype，核对上述结果。
源码核对`fit_model`/`predict_next`/保存字段；`git diff --check`通过。
Current Task=S4.2（Explain待补充，未标Learning Mastered）；Next Task=S4.3仅计划，不自动推进。
无安装/下载、commit/push。待用户用自己的话确认上述细节，无需重跑已完成实验或重答第3–5项。

## S4.2 learning confirmation — 2026-10-20

用户补充Explain准确确认：单例learned(3,2)→手工bank(5,3,2)→16例汇总(16,5,3,2)；
AR=(16,5,4,2)float64、truth同shape为float32，均含初态；TF=(16,5,3,2)float64不含初态，
正确区分递归预测前态与真实前态，状态单位m。
明确phi=[(x−mu_train)/sigma_train,1]，delta=phi C，C为(5,2)共10系数，
与2×4矩阵+bias等价但源码存储参数化不同；30train600/10validation200，整episode隔离，
normalizer与coefficients仅train拟合。结合此前正确第3–5项和实验/预测完成报告，
README及学习包Run/Modify/Explain全部勾选，roadmap同步Learning Mastered。
启动pwd正确，保留已有四份tracked文档修改及三份untracked S4.2文件；本轮仅更新四份状态文档。
未重跑Python实验、训练、安装或下载；环境与工程证据沿用S4.2记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU）。
验证：`git diff --check`通过，`git status --short`核对既有工作保留。
Current Task=S4.2（Engineering Complete + Learning Mastered）；Next Task=S4.3 — Risk / Success Scoring仅计划。
本节仍只有预测对照，无候选选择/闭环改善证据；未自动推进S4.3，未commit/push。

## S4.3 Engineering — 2026-10-20

用户明确请求继续S4.3。启动pwd正确，保留已有README/handoff/roadmap/S4.1 package四份修改与
S4.2 package/script/tests三份untracked文件。读取AGENTS/README/handoff/roadmap、S0.1模板、
S4.2学习包/script/tests与产物contract。本轮只实现S4.3，不进入S4.4。
[学习包](vla/s4_3_risk_success_scoring.md)、[script](../scripts/s4_3_risk_success_scoring.py)、
[tests](../tests/test_s4_3_risk_success_scoring.py)。

重新检查repo `.venv/bin/python` Python3.12.3、NumPy2.5.4及实际模块
`.venv/lib/python3.12/site-packages/numpy/__init__.py`，CPU；无安装/下载/重新训练。
严格加载冻结S4.2 model/AR/truth，检查model/proposal来源hash、schema/plant/action语义、
shape/dtype/clock、AR重算及两条件truth重放；S4.2与其S4.1 source文件前后hash不变。
目标从当前RGB标记+原literal语言按已知camera完美解析，不读真实未来/goal标签，明确不归功于VLA能力。

Risk margin=.02m−未来h1…3到圆心[.015,−.09]m的最小L2距离；
success margin=.01m−末h3到goal的L∞距离。两者独立、边界计入，初态不参与risk，
仅采样点监测，不含连续segment碰撞、contact或plant约束；监测几何不改变ideal-point动力学。
Margin是m，不是概率；flags仅诊断，不选候选/reject/fallback/replan。

64calibration reset groups(seed431)×K5，32heldout reset groups(seed432)×K5，H3/dt.1s。
Probe先定initial/goal，再生成goal-directed noisy commands，非VLA输出；raw输入完整保存。
Calibration/ID plant gain为每group共享的独立uniform[.8,1)，另对相同heldout contexts做gain.65挑战。
校准取每group最坏absolute margin error，再q90 higher(64项排序第58项)，两个buffer各自固定。
Heldout/VLA labels不参与校准；group coverage与event confusion独立报告，不声称概率或正式coverage保证。

Executed during development:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_3_risk_success_scoring.py --output tmp/s4_3_verified > tmp/s4_3_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s4_3*.py'
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_3_risk_success_scoring.py --output tmp/s4_3_final > tmp/s4_3_final_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_3_risk_success_scoring.py --margin-scale 0 --output tmp/s4_3_scale0_final > tmp/s4_3_scale0_stdout.json
```

Final source-provenance artifacts and validation:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_3_risk_success_scoring.py --output tmp/s4_3_final_checked > tmp/s4_3_final_checked_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_3_risk_success_scoring.py --margin-scale 0 --output tmp/s4_3_scale0_checked > tmp/s4_3_scale0_checked_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pip check
git diff --check
```

Risk/success buffers=5.475521/15.516195mm。
Heldout ID risk真实24positive/136negative，raw FP/FN=0/1，margin1=6/0；
success32positive/128negative，raw6/10，margin1=0/32。
Heldout gain.65 risk23/137，raw2/2、margin1=7/0；success42/118，raw23/37、margin1=0/42。
Success buffer大于10mm容差/score上界，故所有calibrated success flags必为False，
与constant_negative基线相同；保留这一失败，不为好看而用heldout调参。
VLA nominal/gain.65每80候选，risk分别56/24与48/32真实正负例，FP/FN分别0/0与8/0，
两套threshold本次计数相同；success正例均0，FNR=null而不是0。
VLA原18次输入16supported/2unknown拒绝仍完整记录；不把输入拒绝称risk拦截，不伪造score。

Group error-buffer覆盖risk/success：calibration58/64、58/64；heldout ID29/32、31/32；
heldout gain.65均8/32；VLA nominal均16/16、shift为0/16与10/16（16为contexts，仅8独立layouts）。
覆盖率不是flag准确率，shift下risk本次0FN不构成泛化保证。
最终batch80 scoring median/P95 .0176/.0290ms，30测量/5warmup，仅geometry+flags；
不含prediction、goal decode、plant、IO、sensor/通信/执行器，不能证明整机实时。
无新controller，closed_loop_success=null；原S4.1 candidate0闭环0/16不被本节chunk标签替代。

6项新增tests通过，全45项tests通过：未来离散几何/初态排除/非连续碰撞、risk与success独立、
group q90与heldout隔离、scale方向/边界、空正类null与全分母、当前观测goal解析、probe独立/可重现。
独立inline重算几何公式、group residual第58项、TP/FP/FN/TN与常量基线；
跨scale source/code hash、scores/labels/raw probes/buffers不变，flags单调且α0等于raw规则。
额外已有output（report未覆盖）、NaN/negative scale、缺失source拒绝通过，不创建无效output。
NPZ逐字段round-trip、修改文档local links、pip check、git diff --check通过。
所有产物ignored tmp，无新依赖/旧repo修改/commit/push。
README/roadmap同步Engineering Complete，S4.3 Learning Run/Modify/Explain均未勾选。
Current Task=S4.3；Next Task=S4.4 — Rejection / Alternate Action仅计划。
用户读Problem→Mathematics→Run→先预测margin-scale1→0后Modify→五问Explain；STOP等待用户确认及选择。

## S4.3 learning confirmation — 2026-10-20

用户明确确认实验和预测均完成；五项Explain准确覆盖risk/success几何margin公式、m单位、
边界/未来h1…3/末h3定义、两事件独立及非连续碰撞限制；scorer只用AR/current goal，
完美合成RGB/literal解析非learned grounding，AR/truth/goal/score/flag/label的精确shape/dtype，
TF不参与评分、未来只作校准/事后评价。
正确计算64×5=320 calibration、32×5=160 heldout与16×5=80旧VLA候选，
group max再q90 higher第58项、buffer5.476/15.516mm，heldout/VLA不参与拟合。
准确说明alpha1→0原始数据/score/label/buffer/coverage不变，risk FP不增FN不减，
success FP不减FN不增及两heldout条件的实际混淆计数；success score上界10mm低于buffer，
alpha1全负与constant_negative等价。正确区分VLA正例0的FNR=null、group error-buffer coverage
与分类准确率/概率校准，以及flags与控制执行、局部latency与整机时延。
README和学习包Run/Modify/Explain全部勾选，roadmap同步Learning Mastered。
启动pwd正确、git status --short为空；本轮仅更新四份状态文档，未重新运行实验、训练、安装或下载。
环境与工程结果沿用S4.3记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU）。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对仅四份文档修改。
Current Task=S4.3（Engineering Complete + Learning Mastered）；Next Task=S4.4 — Rejection / Alternate Action仅计划。
保留success全负退化、机制偏移覆盖下降与无新闭环证据的限制；未自动推进S4.4，未commit/push。

## S4.4 Engineering — 2026-10-20

用户明确请求继续S4.4。启动pwd正确，保留S4.3学习确认遗留README/handoff/roadmap/package四份修改。
读取AGENTS/README/handoff/roadmap、S0.1模板、S4.3 package/script/tests与source产物；
当前S4.4此前无包。本轮只实现S4.4，未进入S4.5/S4.6。
[学习包](vla/s4_4_rejection_alternate.md)、[script](../scripts/s4_4_rejection_alternate.py)、
[tests](../tests/test_s4_4_rejection_alternate.py)。

环境重新确认repo `.venv/bin/python` Python3.12.3、NumPy2.5.4，实际模块
`.venv/lib/python3.12/site-packages/numpy/__init__.py`，CPU；无安装/下载/训练/重新校准。
默认读取`tmp/s4_3_final_checked`；加载其S4.2/S4.1源文件并检查schema/geometry/plant/frame/unit/dt、
model/proposal hashes、AR/truth/scores/labels及校准buffer重算；所有source前后hash一致。
只读继承S4.3 buffers=5.475521/15.516195mm与scale1。
顺手纠正S4.3 goal契约表的dtype表述为probe float32 / VLA float64，与实际NPZ一致，未改变数值或评分语义。

先向用户说明默认risk-only gate与strict对照，避免把S4.3全负success藏掉。
默认拒绝risk flags；candidate0通过即保留，否则在eligible中argmax predicted success margin，
并列最低index，无候选输出−1 sentinel与显式(3,2)float32 zero commands。
strict额外要求success flag。Selector只读predicted scores/buffers/scale，无actual labels/truth/gain。
独立对照candidate0/reject_only/alternate，每条件各G个attempt，分别reset后执行完整H3。
zero不被默认视为安全；unknown文本原2次input拒绝不进gate、不造zero episode。
实际bad=risk事件，good=risk-free且h3 goal success，另报risk-free被拒（包括未达goal的neutral动作）。
全bank分母G×5、primary分母G与实际每方法G个attempt分别记录。

Executed:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_4_rejection_alternate.py --output tmp/s4_4_verified > tmp/s4_4_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_4_rejection_alternate.py --require-success --output tmp/s4_4_strict_verified > tmp/s4_4_strict_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s4_4*.py'
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_4_rejection_alternate.py --output tmp/s4_4_final > tmp/s4_4_final_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_4_rejection_alternate.py --require-success --output tmp/s4_4_strict_final > tmp/s4_4_strict_final_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pip check
git diff --check
```

Default全候选计数：probe ID160中actual bad24全拒，risk-free136中误拒6，actual good26中误拒2；
probe gain.65的160中bad23全拒，risk-free137中拒7，good30中误拒0。
VLA nominal80中bad56全拒/risk-free24拒0；shift80中bad48全拒/risk-free32拒8；good均0。
allowed_bad本次均0，但不保证随后fallback安全。Primary子集统计独立见report。

实际H3执行unsafe/risk-free goal successes：probe ID candidate0=5/1、reject_only=3/1、alternate=3/1，
每方法32attempts；probe gain.65为4/4、3/4、3/4。Alternate两probe条件均27primary/1alternate/4fallback，
各有3次zero fallback仍在监测圆盘内（初态已在圆盘），保留此失败，不加escape oracle。
VLA两条件每方法16attempts，candidate0 unsafe16、goal success0；reject_only zero16次unsafe0/success0；
alternate替换16次unsafe0/success0，没有任务成功改善证据。

Strict因为success score≤10mm<15.516mm buffer，所有候选被拒，probe每条件32/32 fallback、
VLA每条件16/16 fallback；probe仍unsafe3/32、goal success0，VLA unsafe0/16、success0。
Strict误拒probe全部26/30个good；VLA good为0，误拒good0不代表完美保留有用动作。
原source/model/buffers/raw bank/scores/labels/gain及candidate0 baseline不变。

Gate-only batch latency：probe ID median/P95 .0221/.0231ms（32contexts），
VLA nominal .0645/.0787ms（16contexts），均30测量/5warmup。
仅flags+candidate选择，排除WM/score计算/commands/plant/IO/真实链路，不作速度排名或实时保证。
只做一次选择后H3 open-loop replay，不读执行反馈重规划，closed_loop_success=null。
S4.1原8步candidate0闭环0/16能力证据不被本节三步结果覆盖。

7项新增测试通过，全52项通过：primary优先、eligible argmax/tie、无备选与−1显式zero、
strict全fallback、risk-free与good分开、zero仍unsafe、selector不读实际标签及NaN拒绝。
独立inline跨strict核对source/code hash、原始数据/score/labels不变量，重算gate与chosen、
逐条commands→float32 plant三步、T/T+1/dtype/clock、risk/goal及所有方法分母/stage分区。
NPZ逐字段round-trip、已有output保护、missing source与unit错配拒绝、本地Markdown链接、
pip check、git diff --check通过。产物ignored tmp，未修改旧repo、commit/push。
README/roadmap同步S4.4 Engineering Complete，Learning Run/Modify/Explain未勾选。
Current Task=S4.4；Next Task=S4.5 — Receding-horizon Replanning仅计划。
用户读Problem→Mathematics→Run→先预测require-success后Modify→五问Explain；STOP等待用户确认及选择。

## S4.4 learning confirmation — 2026-10-20

用户明确确认实验和预测均完成；五项Explain准确说明risk-only/strict拒绝公式、primary-preserving选择、
eligible success argmax/稳定tie、无候选−1显式zero且不能bank[-1]；selector仅读预测margin/冻结buffer/
scale与开关、不读truth/labels/gain，bank/score/rejected/index/command/actual trajectory/clock精确契约。
正确区分bad、risk-free、good，160/80候选与32/16每方法attempt、primary子集分母及误拒risk-free/good；
解释allowed_bad=0不包含新增fallback，probe3个初态在圆盘内zero仍unsafe，VLA监测risk降低但goal0/16。
明确strict改变gate/选择/实际执行，原model/source/bank AR/scores/buffers/bank truth/gain与baseline不变；
success score上界10mm低于buffer15.516mm导致全fallback与误拒所有26/30个good，不构成安全/控制提升。
README与学习包Run/Modify/Explain全勾选，roadmap同步Learning Mastered。
启动pwd正确、git status --short为空；本轮仅更新四份状态文档，未重跑实验、训练、安装或下载。
环境与工程证据沿用S4.4记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU）。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对四份文档修改。
Current Task=S4.4（Engineering Complete + Learning Mastered）；Next Task=S4.5 — Receding-horizon Replanning仅计划。
保留zero fallback风险、strict退化及无新闭环证据的限制；未自动推进S4.5，未commit/push。

## S4.5 Engineering — 2026-10-20

用户明确请求继续S4.5。启动pwd正确，保留S4.4学习确认已有README/handoff/roadmap/package四份修改。
读取AGENTS/README/handoff/roadmap、S0.1模板、S4.4学习包/script与S3.7反馈源码，
核对S1.7 checkpoint loader/S4.1 proposal接口；S4.5此前无包。本轮仅实现当前任务，未进入S4.6。
[学习包](vla/s4_5_receding_horizon.md)、[script](../scripts/s4_5_receding_horizon.py)、
[tests](../tests/test_s4_5_receding_horizon.py)。

环境重新确认repo `.venv/bin/python` Python3.12.3 / NumPy2.5.4，实际NumPy模块
`.venv/lib/python3.12/site-packages/numpy/__init__.py`，CPU；无安装/下载/训练/重新校准。
默认source `tmp/s4_4_final`，完整沿S4.4→S4.3→S4.2→S4.1→S1.5读取与hash/语义核对；
model/weights readonly，source前后hash一致，继承原risk-only、alpha1与buffers。
每decision从当前RGB/text/observed-or-belief state重新提K5/H3，冻结WM预测后gate，执行min(E,8−t)，
丢弃旧后缀；没有warm suffix或候选搜索扩张。H3预测/gate始终固定，包括episode末预算外未执行尾部，
明确记录这一额外保守/rolling success与episode goal不同的限制。
Observed分支在decision边界刷新actual float32 state；belief分支只沿model prefix endpoint转float32推进。
Planning goal从当前RGB/literal解析，evaluator annotation不进控制；gain/kick只在plant。

Protocol四条件各8旧validation layouts×2texts=16attempts：nominal gain1、hidden gain.65、
hidden world-x+.12m kick在第2次transition后、unsupported grab red/blue block。
每run双方法共128attempts，预算8、dt.1s，无early success stop；精确观测/静态RGB，不是真机器人。
未知语言保留T0拒绝/一个初始state/空action，绝不伪造zero episode；原proposal ValueError stage
记为policy_input_or_output_rejection，错误原因写入plan。其它stage为monitoring_risk或final_goal_error_after_budget。
Goal success与risk-free episode success分开，实际监测风险看执行p1…p8，不看初态/连续线段。

Executed:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_5_receding_horizon.py --output tmp/s4_5_verified > tmp/s4_5_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_5_receding_horizon.py --execute 3 --output tmp/s4_5_e3_verified > tmp/s4_5_e3_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s4_5*.py'
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_5_receding_horizon.py --output tmp/s4_5_final > tmp/s4_5_final_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_5_receding_horizon.py --execute 3 --output tmp/s4_5_e3_final > tmp/s4_5_e3_final_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pip check
git diff --check
```

两E/双方法所有组success与goal success均0/16，已执行unsafe均0/16；前三条件完整8步、
各16个final_goal_error_after_budget；unsupported各16次输入拒绝，正常accepted latency与RMSE为null。
E1每episode8plans、完整16case组128accepted；E3每episode3plans(ticks0/3/6)、prefix3/3/2、组48accepted。
本次fallback0，单元测试strict模拟覆盖fallback执行/再次重规划；不据此否定S4.4 fallback风险。
每run5次完整episode warmup排除；原source/hash/protocol与跨E首decision bank/WM/chosen不变。

执行前缀RMSE[m] observed/belief E1：nominal2.794e-9/6.202e-9、gain.012354/.043198、kick.030000/.079373。
E3：nominal2.678e-9/4.261e-9、gain.014478/.030785、kick.042426/.079373。
仅已执行prefix、完整16case组256个物理coordinate scalars；belief误差含历史漂移，跨E也改变lead time与visited states。
Kick解析核对：E1仅step2带120mm x误差→120/sqrt16=30mm；E3 step2/3带误差→42.426mm；
belief step2…8七个位置均带偏差→79.373mm。真实反馈修正下一规划起点，但没有改善任务成功。

Nominal observed planning per-call median/P95 E1 .2525/.3550ms(128calls)、E3 .2673/.4577ms(48calls)；
per-episode planning median E1 2.0754ms / E3 .8056ms。
每decision分别render/proposal/WM/score_gate、planning总和、plant apply；总和不再与子项相加。
Score_gate含goal解析/margins/gate/gather，fallback时含zero WM预测；输入拒绝另计，不混正常推理。
全部accepted本地compute超过n*.1s预算次数0；无sensor/通信/servo/IO等待，也没有按模拟时钟sleep，非整机实时保证。

7项新增tests与全59项通过：E-prefix/末余2、kick观察边界、belief不读gain/kick实测、evaluator goal隔离/无early stop、
strict zero持续执行/重规划、unknown零动作保留、prefix RMSE真实分母。
独立inline审计两run共256attempt的gain/kick逐转移、T/T+1/clock/dtype、所有prefix、feedback/belief起点、
belief跨nominal/gain/kick的bank/prediction/commands相同、跨E源hash/首decision不变、goal/risk/失败/256scalar分母。
NPZ逐字段round-trip、已有output保护、E0/E4/缺source拒绝、文档local links、pip check、git diff --check通过。
产物ignored tmp；未修改旧repo，未commit/push。
README/roadmap同步S4.5 Engineering Complete，Learning Run/Modify/Explain未勾选。
Current Task=S4.5；Next Task=S4.6 — Final Benchmark仅计划。
用户读Problem→Mathematics→Run→先预测E1→3后Modify→五问Explain；STOP等待用户确认及选择。

## S4.5 learning confirmation — 2026-10-20

用户明确确认实验和预测均完成；五项Explain准确说明当前RGB/text/planning state→K5/H3→WM/gate→
执行min(E,8−t)的数据流及shape/dtype/单位，observed实测与belief预测prefix endpoint来源、
plant gain/kick和evaluation goal隔离；E1 ticks0…7与E3 ticks0/3/6、prefix3/3/2、kick t2/t3反馈，
固定H3末尾评分到t9/t10与丢弃旧后缀，不混同remaining budget。
正确解释belief跨nominal/gain/kick命令不变、120mm扰动影响1/2/7状态，256scalar分母和
30/42.426/79.373mm误差推导；128attempt、16case分组、128/48accepted plans与T0拒绝全分母、null/stage。
明确unsafe0但goal/success0不证明控制能力、stage非根因；跨E模型/hash/normalizer/bias/H/K/dt/gate/protocol/
t0选择不变量，正确解释latency分段包含关系、每episode compute及本地overrun0、旧probe buffer/
精确proprio/完美goal解析/旧布局的证据限制。
README与学习包Run/Modify/Explain全勾选，roadmap同步Learning Mastered，保留策略失败结论。
启动pwd正确、git status --short为空；本轮仅更新四份状态文档，未重跑实验、训练、安装或下载。
环境与工程证据沿用S4.5记录（repo .venv Python3.12.3 / NumPy2.5.4 CPU）。
验证：`git diff --check`通过；`git diff --stat`与`git status --short`核对四份文档修改。
Current Task=S4.5（Engineering Complete + Learning Mastered）；Next Task=S4.6 — Final Benchmark仅计划。
未自动实施最终benchmark，未commit/push。

## S4.6 Engineering — 2026-10-20

用户明确请求继续S4.6学习。启动pwd正确，保留S4.5学习确认已有README/handoff/roadmap/package四份修改。
读取AGENTS/README/handoff/roadmap、S0.1模板、S4.5 package/script/tests、S1.5数据split生成逻辑与依赖。
本轮只实现S4.6，未扩张训练/调参/新阶段。[学习包](vla/s4_6_final_benchmark.md)、
[script](../scripts/s4_6_final_benchmark.py)、[tests](../tests/test_s4_6_final_benchmark.py)。

环境重新确认repo `.venv/bin/python` Python3.12.3 / NumPy2.5.4，实际模块
`.venv/lib/python3.12/site-packages/numpy/__init__.py`，CPU；无安装/下载/训练/重校准。
默认`tmp/s4_5_final`，沿完整来源链加载readonly VLA/WM/统计/gate，前后hash一致。
来源校验发现S4.5记录相对路径而gate来源使用绝对路径；比较前resolve规范化路径键，
仍逐文件严格比较内容hash，没有更换source或放松检查。

Matched baseline只生成K1 learned candidate0且不调用控制用WM/goal decoder/gate；
WM系统K5包含相同candidate0+四手工bias，附加frozen affine WM/完美当前RGB目标解析/gate。
同case/reset/RGB/text接口/actual-state反馈/E/H3/dt.1/budget8/plant gain-kick与评价定义，
但visited states可不同、额外组件也不同，不能把差异仅归因于WM。一次forward每decision，无oracle控制。
控制计时结束后才做baseline shadow WM或WM full-H3反事实plant，诊断结果不回流control；
完整H3可含预算外尾部，rolling good不等于episode success。

协议：从240个grid ordered red/blue pairs中排除S1.5 train32+validation8。
固定seed461选8、seed462从剩余pool选8，两seed无重叠，共16新layout groups。
每layout两texts×nominal/gain.65/kick/unsupported四条件=128pairs，双methods共256attempts/run。
协议执行前保存并固定SHA256，未据结果换seed/阈值。新pair仍使用旧grid/颜色/camera/语言，非开放世界泛化。
每seed/condition/method16attempts，合并seed32；条件复用16layouts，不把所有attempt视为独立场景。
VLA来源训练metadata实际读取seed7/2400steps；WM data421/split422、calibration431；protocol seeds不是训练seed重复。
每method五完整episode warmup排除；测量case内运行顺序交替。

Executed:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_6_final_benchmark.py --output tmp/s4_6_verified > tmp/s4_6_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_6_final_benchmark.py --execute 3 --output tmp/s4_6_e3_verified > tmp/s4_6_e3_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p 'test_s4_6*.py'
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_6_final_benchmark.py --output tmp/s4_6_final > tmp/s4_6_final_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_6_final_benchmark.py --execute 3 --output tmp/s4_6_e3_final > tmp/s4_6_e3_final_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_6_final_benchmark.py --output tmp/s4_6_final_checked > tmp/s4_6_final_checked_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_6_final_benchmark.py --execute 3 --output tmp/s4_6_e3_checked > tmp/s4_6_e3_checked_stdout.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pip check
git diff --check
```

两个E各256attempts，其中192完整8步轨迹、64个T0拒绝（每method32），原始state/action/clock不造假。
每condition/method32分母：E1两系统success均0；baseline nominal/gain unsafe32/32，WM0/32；
kick两系统unsafe0/success0，unknown输入拒绝均0动作/RMSE与accepted latency null。
E3 nominal/gain同样risk差异、success0；kick baseline1/32 success，WM0/32。
唯一成功为seed461、case `s461_kick_layout_166_0`，final L∞ goal error.004257508m；
seed461该组1/16，seed462为0/16。保留paired反例，不称WM改善success，也不以单例声称baseline鲁棒。
Nominal/gain baseline stage monitoring_risk，WM stage final_goal_error_after_budget；kick主要goal miss，unknown为input/output拒绝。

E1 WM每条件256decisions×5=1280 candidate exposures：nominal/gain/kick actual bad144/165/16均被拒，
risk-free被拒98/104/187，actual good1/0/5、false rejected good全0。
Primary子集各256：拒绝57/73/57，actual primary bad32/32/0全拒，risk-free primary拒25/41/57。
Good稀少或0，不能把0误拒推广为有用性/召回保证；allowed rolling good不代表选中或episode成功。
E3每条件96decisions×5=480 exposures，bad139/132/16全拒，risk-free拒11/25/98，good0/0/3、误拒good0。
所有数据都是WM own-visited-state反事实/重叠H3窗口，非独立1280场景。本次fallback0，不推翻S4.4 fallback失败。

执行prefix RMSE[mm] baseline shadow/WM：E1 nominal≈2.172e-6/3.294e-6、gain12.624/12.173、kick30/30；
E3 nominal≈2.586e-6/2.695e-6、gain20.292/14.413、kick42.426/42.426。
完整32case组分母512physical scalars，unknown分母0→null；跨系统visited states不同，不作共同输入模型排名。

最终checked nominal controller median/P95[ms]：E1 baseline.1798/.4876(256calls)、WM.3130/.6959(256)；
E3 baseline.1954/.4471(96)、WM.3432/1.2565(96)。Per-episode control median分别1.4309/2.5519ms与.5912/1.0630ms。
控制total含render/proposal/可选WM/gate/gather；baseline WM/gate=null，shadow/oracle diagnostics与toy apply单列。
所有accepted本地control overruns0，但排除真实sensor/transport/servo/IO，不证明整机实时或多轮性能稳健性。

7项新增tests与全66项通过：新pair/split/seed隔离、baseline controller不读WM/goal decoder、
shadow模型改变不影响baseline动作、evaluator goal隔离、counterfactual kick对齐、matched E/primary、T0/缺pair拒绝。
独立inline审计两E共512attempt：16新groups隔离、case pairing、逐步gain/kick、prefix/float32/clock、
shadow SSE/反事实rejection、512scalar与seed/失败分母、跨E protocol/source/t0不变量。
最终checked输出与此前完整审计raw轨迹逐值相同，paired/protocol一致且protocol hash复核通过。
NPZ逐字段round-trip、已有output/E0/E4/missing source拒绝、local Markdown links、pip check、git diff --check通过。
所有产物ignored tmp，无旧repo修改、commit/push。
README/roadmap更新Stage0–4全部Engineering Complete，S4.6 Run/Modify/Explain未勾选。
Current Task=S4.6；Next Task=无自动后续任务，等待S4.6学习确认与用户选择。
用户读Problem→Mathematics→Run→先预测E1→3后Modify→五问Explain；STOP。
最终工程结论：附加WM系统在本协议降低部分离散监测risk、增加compute，未提升任务成功；
总体tiny策略能力仍未达标，不能声称WM单独贡献/真实机器人/广泛OOD/安全或大型foundation模型能力。
