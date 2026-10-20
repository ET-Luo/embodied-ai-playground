# S4.6 — Final Benchmark

## Problem

固定checkpoint、协议、动作语义、执行预算和结果定义，对比VLA-only与VLA+WM。
报告closed-loop success、failure stage、执行预测误差、rejected bad/误拒good及controller latency。
保留失败、拒绝、配对反例和附加资源差异，不以“加入WM”直接推断成功改善。

## Intuition

两组从同一状态出发、面对同一场景/指令/扰动，按同一E和8步预算行动，才能比较系统结果。
控制之后轨迹可以不同；要比较的是相同case下的结果，不是强迫两组始终访问相同状态。
更少的监测风险与更高的任务成功率是不同指标；系统也可能用更多计算却仍到不了目标。

## Input / Output Contract

[script](../../scripts/s4_6_final_benchmark.py)，[tests](../../tests/test_s4_6_final_benchmark.py)。
默认`--source tmp/s4_5_final`，校验S4.5 schema/来源hash，再沿S4.4→S4.3→S4.2→S4.1→S1.5加载。
相对/绝对路径解析为同一文件后比较来源hash，防止仅键名不同而误判；不放松内容一致性检查。
权重、WM、normalizer、buffers与gate冻结，无训练/下载/重校准。

| 项 | shape / dtype | 单位、frame、时序与模块流向 |
|---|---|---|
| current RGB/text/state | (32,32,3) uint8 / string / (2,) float32 | 静态合成camera、literal语言、exact measured world XY[m] |
| VLA-only bank | (1,3,2) float32 | 原冻结learned candidate0命令[m]，不构造四个额外bias |
| VLA+WM bank | (5,3,2) float32 | 相同candidate0 + 原S4.1四个手工偏置，逐分量±.06m |
| WM future | (K,4,2) float64 | 当前实际状态+AR3 world XY预测[m]；gate固定H3 |
| margins / rejected / chosen | (5,) float64[m] / (5,) bool / int | 仅WM组用于控制；−1为显式zero |
| actual state/action | (T+1,2)/(T,2) float32 | measured world XY / commanded world XY displacement[m]，dt=.1s |
| clock | (T+1,) float64 | 模拟秒，正常T8；input rejection T0，不造假命令 |
| evaluator bank futures | (5,4,2) float32 | 从WM组当前实际state独立重放gain/kick，**不进入控制** |

输出`contract.json` schema `paired_vla_wm_benchmark/1.0`、先行固定的`protocol.json`、
`executions.npz`完整raw state/action/clock、`report.json`逐decision记录/paired/per-seed汇总、`summary.csv`。
NPZ读取allow_pickle=False并逐字段round-trip；protocol hash和所有source hashes运行后复核。

## Core Concepts

- VLA-only使用原candidate0，刷新真实state、同E执行，没有控制用WM、goal decoder或gate。
- VLA+WM在相同观测接口上额外使用四个手工bias、WM与完美当前RGB goal decoder。
  这是**两套完整系统**的matched对照，不能把差异只归因于learned WM。
- 两组同checkpoint、reset/case、RGB/text接口、E/H3/dt/预算/plant扰动/评价定义；后续visited state不同。
- 每matched case独立reset；两组运行顺序随case奇偶交替，减轻固定顺序计时偏差。
- 原S4.5旧8validation布局不作为本节新test；新增配对排除S1.5全部32train+8validation。
- 新配对仍在同16格坐标、同颜色/renderer/语言上，不是新物体、连续位置、camera或真实机器人泛化。
- Benchmark前保存完整协议并固定hash；不根据结果选择新seed、训练checkpoint或调整阈值。

## Mathematics

### Frozen protocol

16个grid positions构成16×15=240个有序red/blue pairings。
排除S1.5 40个train/validation pairs；seed461从剩余200中无放回选8，
seed462从排除已选后的192中无放回选8。两seed共16个不重叠layout groups。
每layout两颜色指令、四条件，128 matched cases；每case两个methods，**256 attempts/run**。
每seed/condition/method=16attempts，合并seed后每condition/method=32attempts。
协议seed不是模型训练seed：本来源VLA训练seed7/2400steps，WM data421/split422，buffer calibration431。
VLA训练metadata实际从来源report读取；两个protocol seeds不等于两个独立训练模型。

### Closed-loop outcome

两组按actual p_t重提案，执行n=min(E,8−t)；H3始终固定，丢弃旧后缀。
Plant：p_next=p+gain·command+kick，gain1或.65；kick组在transition2后world x+.12m。
Success=[完成8步 AND final L∞ goal error≤.01m AND 未触发任何执行后采样点圆盘risk]。
Goal success独立报告；没有early stop，风险发生也继续记录后续轨迹。
失败stage：policy_input_or_output_rejection；monitoring_risk优先；否则final_goal_error_after_budget。

### Rejection quality

仅WM组，在**其自己的visited actual state**，对5条原bank做完整H3反事实plant重放。
保留全H3，包括预算外未执行尾部；gain/kick只给evaluator，反事实labels从不回流selector。
Bad=[actual full-H3 risk]；Good=[无risk且rolling H3末点到goal≤.01m]。
这不是episode t8 success，也不代表被允许good一定会被选择。
分别统计全bank与candidate0 primary的rejected_bad/allowed_bad/rejected_risk_free/false_rejected_good。
分母是**decision-candidate exposures**，不是独立轨迹/episode数；相邻decision的H3窗口还会重叠。

### Prediction error and latency

只比较实际执行prefix的状态误差，RMSE=√(Σerror² / (2·实际commands总数))。
合并seed的完整32case组分母32×8×2=512；unsupported组分母0，RMSE=null。
VLA-only预测来自控制计时结束后的**evaluation-only shadow WM**，不改变动作/反馈状态或控制耗时。
两组用相同冻结模型在各自轨迹上诊断，visited states/动作不同，不能把两数作公平模型准确率排名。

Control total包括render+proposal+可选WM+gate/gather；shadow/oracle诊断、toy apply分别计时。
没有控制用WM的baseline对应WM/gate latency为null，不是“计入后再减去”。
Unsupported拒绝另计时，不混accepted median/P95；controller和子阶段不能重复相加。

## Architecture / Data Flow

冻结来源→新pairing协议预先存盘→matched case独立reset→当前RGB/text/actual state。
分支A：frozen VLA candidate0→prefix执行。
分支B：same VLA+biases→WM→current RGB goal/margins→gate→candidate/zero→prefix执行。
两分支均反馈actual state，直到8步或输入拒绝。
独立evaluation在controller stopwatch之后做shadow/反事实labels，收集raw/paired/per-seed metrics。

## Math / Architecture → Code

`fresh_protocol`排除旧40pair、两seed无重叠地选新16pair。
`control`显式分开baseline与WM路径，不接plant/evaluation annotation。
`actual_future`仅为full-H3反事实标签，按global tick处理kick。
`run_episode`控制计时截止后做诊断，实际执行相同plant规则；`summarize`保留全部分母与null，
`paired_counts`对同case两结果计数；`main`先写protocol、每method五episode warmup排除、交替顺序运行。

## Minimal Experiment

CPU repo .venv / NumPy，无新依赖/安装/下载/训练。两条使用相同S4.5 source，第二条先预测E1→3：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_6_final_benchmark.py --output tmp/my_s4_6
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_6_final_benchmark.py --execute 3 --output tmp/my_s4_6_e3
```

自有产物两条均加`--source tmp/my_s4_5`。原S4.5 source还须保有完整上游来源链。
此处实测继承risk-only/alpha1 gate；不同source不能直接套用数字。
看protocol/contract/report.json、executions.npz、summary.csv；重点检查paired反例和seed拆分。

## Expected Result

新pairing必须与旧40pair及另一seed无交集，两methods同case/预算/E/初始candidate0。
WM系统可能减少监测risk但付出更多compute、误拒risk-free，并未必提高success。
E1→3不改变protocol/参数/阈值/t0决策，却改变后续trajectory/窗口计数/结果；不保证成功单调。

## Actual Result

2026-10-20，repo .venv Python3.12.3 / NumPy2.5.4 CPU，模块在本repo .venv。
两个E设置各256attempts=128pairs，完整case192条8步轨迹与64条T0拒绝；
所有条件间复用同16layout groups，不能当256独立场景。
每行分母32；success是包含risk要求的episode success，本次goal_success计数与其相同：

| condition | E1 VLA-only success / unsafe | E1 VLA+WM success / unsafe | E3 VLA-only success / unsafe | E3 VLA+WM success / unsafe |
|---|---:|---:|---:|---:|
| nominal | 0 / 32 | 0 / 0 | 0 / 32 | 0 / 0 |
| gain.65 | 0 / 32 | 0 / 0 | 0 / 32 | 0 / 0 |
| kick | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 |
| unsupported | 0 / 0（T0拒绝） | 0 / 0（T0拒绝） | 0 / 0（T0拒绝） | 0 / 0（T0拒绝） |

E1 paired nominal/gain各32对只在VLA-only发生risk，WM-only发生risk为0。
E3 kick出现VLA-only独有success一例：seed461、layout166、red，final L∞ error4.258mm；
同一case WM系统失败。seed461该组1/16，seed462为0/16，合并1/32。
原样保留这一反例，不称WM提高了success，也不以单例推断显著性或baseline已具备鲁棒能力。

Failure stages：nominal/gain baseline各32个monitoring_risk；WM对应各32个final_goal_error_after_budget。
Kick E1两组各32个goal miss；E3 baseline31个goal miss+1success，WM仍32个goal miss。
Unsupported每method32个policy_input_or_output_rejection，只有初态/空action，RMSE/accepted latency=null。

### Rejected actions（E1 WM组）

每条件256decisions×5=1280 exposures，primary子集分母256；没有把它们当1280独立场景。

| condition | actual bad / rejected bad | actual risk-free / rejected risk-free | actual good / false rejected good |
|---|---:|---:|---:|
| nominal | 144 / 144 | 1136 / 98 | 1 / 0 |
| gain.65 | 165 / 165 | 1115 / 104 | 0 / 0 |
| kick | 16 / 16 | 1264 / 187 | 5 / 0 |

Primary被拒57/73/57次，其中actual primary bad32/32/0，risk-free primary被拒25/41/57。
Kick primary全risk-free却仍被拒，说明名义WM对未来kick不知情加上buffer产生保守拒绝。
本次误拒good0，但good支持极少/为0，不能称完美召回或有用性；所有WM episode仍失败。
E3每条件96decisions×5=480 exposures，bad及rejected bad139/132/16；
risk-free被拒11/25/98，good0/0/3且false rejected good均0。分母变化与visited states变化一起解释。
本次没有fallback calls，不能推翻S4.4 zero可能仍unsafe的反例。

### 执行前缀prediction RMSE mm

| condition | E1 VLA-only shadow | E1 WM | E3 VLA-only shadow | E3 WM |
|---|---:|---:|---:|---:|
| nominal | .000002172 | .000003294 | .000002586 | .000002695 |
| gain.65 | 12.624 | 12.173 | 20.292 | 14.413 |
| kick | 30.000 | 30.000 | 42.426 | 42.426 |

Shadow预测未参与baseline选择；数值不是offline loss，不代表共同输入下的模型排名。
近零nominal误差来自理想线性系统；gain/kick差异与prefix和访问轨迹有关。

### 本地计算耗时

最终一次run nominal：

| E / system | controller median / P95 ms | calls | per-episode control median ms |
|---|---:|---:|---:|
| E1 VLA-only | .1798 / .4876 | 256 | 1.4309 |
| E1 WM | .3130 / .6959 | 256 | 2.5519 |
| E3 VLA-only | .1954 / .4471 | 96 | .5912 |
| E3 WM | .3432 / 1.2565 | 96 | 1.0630 |

每method五完整episode warmup排除，case内method顺序交替；没有独立系统负载控制或多轮性能稳健性保证。
Controller只计部署动作链，shadow/oracle diagnostics单列；apply仅toy更新，无真实sensor/transport/servo/IO。
所有accepted本地control超过n×100ms预算次数0，不能当实机实时保证或deadline测试。

7项新增tests与全66项通过；独立审计两E共512attempt的layout隔离、pair、全部物理转移/kick、
prefix/dtype/clock、shadow SSE、反事实拒绝计数、seed/失败/512scalar分母、跨E协议/hash/t0不变量。
最终产物与准确命令见handoff；没有删除失败case、调整buffer或更换新seed。

## Result / Explanation

本benchmark支持的结论：冻结小系统在这些新grid pairings中，附加候选/WM/goal解析/gate降低了
名义与gain条件的离散监测risk，增加本地controller compute，未观察到任务成功改善；E3 kick还有反例。
它不支持WM单独因果贡献、机器人安全、开放世界泛化、模型复现或大型foundation能力声明。

## Project Result

Stage0–4工程学习链已覆盖数据契约→tiny VLA→现代架构概念→tiny WM→验证/gate/反馈→matched评价。
原始物理数据、train-only normalization、来源/时序/失败分母与CPU约束均保留。
工程包完成与任务能力达标分开：这次benchmark已完成，learned policy总体仍失败；
S4.6 Learning Mastered（2026-10-20 用户确认Run/Modify/Explain），不自动开始新阶段或修复任务。

## Failure Analysis / Limitations

| 限制 | 本轮证据边界 |
|---|---|
| single checkpoint | 协议seed461/462不是训练seed重复，无multi-training-seed稳健性 |
| 新pair但旧grid | 新16pair不等于新物体/位置/语言/真实camera泛化 |
| added components | biases与完美goal decoder共同变化，不能只归功于learned WM |
| sparse good/success | zero false-good-rejection与单个baseline成功都不能作广泛能力结论 |
| full H3反事实 | rolling labels可超episode预算，不等价于已执行risk或最终success |
| own visited states | 跨方法prediction/rejection输入不同，不能误当公共固定测试集 |
| calibration shift | buffers未在新visited分布重校准，无正式风险/概率保证 |
| sampled geometry | 不含连续碰撞、contact、initial风险或物理安全证明 |
| timing exclusions | shadow/oracle仅评价；本地controller/ms不含真实robot链路或IO |

## Relation to Robotics

通过版本化文件/动作语义连接policy、model、gate、执行和独立evaluator，展示匹配评价的必要边界。
不导入或修改sibling repo；没有真实机器人任务成功声明。

## Interview Capsule / Notes

30秒：我冻结同一tiny VLA与WM，用16个新物体pair、两个协议seed做matched闭环评价。
WM系统降低离散risk但没有success收益，还增加compute；E3 kick有一例baseline独有成功，保留而不筛掉。

2分钟：解释matched接口与不同visited states、额外bias/goal组件、shadow/oracle计时隔离、
曝光计数与episode分母、为何工程完成不等于机器人或foundation能力达标。

## Likely Follow-up Questions

1. 两系统哪些条件匹配，哪些额外组件不同？为什么不能把差异只归因于WM，为什么后续state不必相同？
2. 新layout怎样排除旧40pairs和跨seed重叠？16groups、128pairs、256attempts、每条件32及512scalars分别是什么？
3. Baseline shadow WM与full-H3反事实labels何时计算、用于什么？为什么rolling good和1280exposures不是episode success或独立场景？
4. E1→3哪些协议/模型/t0不变量保留，哪些计数/轨迹/结果会变？如何解释E3 kick baseline1/32、WM0/32和误拒good0？
5. 最终证据支持哪些风险/成功/latency结论，又不能支持哪些泛化/安全/WM因果/多训练seed结论？

## Must Remember

Matched结果≠WM单独贡献；protocol seed≠training seed；拒绝exposure≠episode；保留反例；工程完成≠能力达标。

## My Verification

- [x] Run：检查先行冻结protocol、new-pair隔离、matched结果、seed拆分、拒绝分母和计时边界。
- [x] Modify：先预测E1→3协议不变量、call/exposure变化与success非单调可能，再新目录运行核对。
- [x] Explain：独立回答上述五问，用证据描述项目完成与能力限制。

Engineering Complete + Learning Mastered（2026-10-20 用户确认）。Current Task=S4.6；Next Task=无自动后续任务，等待用户选择。

2026-10-20 用户明确确认实验和预测均完成，五项Explain准确覆盖matched接口与额外bias/WM/goal decoder/gate
组件的归因限制、不同visited states；240−40的new-pair排除与双seed不重叠、16groups/128pairs/256attempts/
每组32/512scalars及T0 null；shadow/oracle在controller计时之后、不回流控制，rolling good与重叠exposures
非episode success/独立场景；跨E不变量与calls/exposures变化、E3 baseline独有成功及good支持稀少；
风险降低但success不增、额外compute与有限grid/single checkpoint/旧buffer/离散监测/无实机IO的证据边界。
Run/Modify/Explain依据用户报告全部勾选；Stage0–4工程与学习完成，策略能力仍未达标。
