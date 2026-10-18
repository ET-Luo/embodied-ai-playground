# P4 project handoff

Updated 2026-10-18 (Europe/Berlin)。新会话读 AGENTS.md → 本文 → README → 当前学习包。

Current Task = **S2.5 — π0 Architecture Trace**，Engineering Complete + Learning Mastered（2026-10-18 用户确认）。
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
Next Task = **S2.6 — π0.5 Study**，仅计划，不授权自动进入。
已实现literal条件state MLP、tiny learned word embedding、独立tiny CNN定位和summary融合对照；已有tiny averaged-patch Transformer action chunk但未学成控制；另有独立tiny语言目标泛化诊断和冻结checkpoint闭环评价，未修复S1.5；没有运行真实OpenVLA或World Model；S2.3仅架构与接口学习。

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
