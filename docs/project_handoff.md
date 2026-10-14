# P4 project handoff

Updated 2026-10-14 (Europe/Berlin)。新会话读 AGENTS.md → 本文 → README → 当前学习包。

Current Task = **S0.7 — Action Chunk Representation**。Engineering Complete + Learning Mastered（2026-10-14 用户确认）。
S0.6 Engineering Complete + Learning Mastered（2026-10-14 用户确认）。
S0.5 Engineering Complete + Learning Mastered（2026-10-14 用户确认）。
S0.4 Engineering Complete + Learning Mastered（2026-10-14 用户确认）。
S0.3 Engineering Complete + Learning Mastered（2026-10-14 用户确认）。
S0.2 Engineering Complete + Learning Mastered（2026-10-14 用户确认）。
S0.1 Engineering Complete + Learning Mastered（2026-10-13 用户确认）。
Next Task = **S1.1 — Language-conditioned State Policy**，仅计划，不授权自动进入。
当前没有 Tiny VLA、OpenVLA、World Model 实现；后续阶段只有 roadmap 与空目录。
Stage0（S0.1–S0.7）Engineering Complete + Learning Mastered；Stage1尚未启动。

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
