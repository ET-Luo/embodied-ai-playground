# S4.4 — Rejection / Alternate Action

## Problem

将S4.3的诊断flag接成真正的chunk gate：拒绝原提案、选允许的替代候选，或输出显式fallback。
分开统计rejected bad actions、误拒risk-free动作和误拒真正good动作，再独立执行所选chunk。
本任务只有一次选择、H3 open-loop执行；不做S4.5的observe/replan或S4.6最终benchmark。

## Intuition

“拦下一个坏提案”不代表之后执行的替代或停止动作一定安全，也不代表任务能成功。
先检查candidate0；合格就保留，不因别的候选评分更高就换掉它。
原提案被拒时，再从允许的候选中挑末点最接近目标的一条；没有允许候选，明确执行三次零命令。
如果点已经在禁入区，零命令会让它继续留在里面，必须把这类失败保留。

## Input / Output Contract

[script](../../scripts/s4_4_rejection_alternate.py)，[tests](../../tests/test_s4_4_rejection_alternate.py)。
默认读取`tmp/s4_3_final_checked`，自有产物用`--scores tmp/my_s4_3`。
还需S4.3记录的S4.2/S4.1源文件可用；缺失报错，不隐式生成/训练/下载。
加载检查schema、plant/frame/unit/dt、圆盘/goal geometry、source/model/proposal hashes，
重算buffer、原AR、原truth和scores/labels。S4.3缓冲量与margin-scale完全沿用，不重新校准。

| 项 | shape / dtype | 单位、frame、时间与模块流向 |
|---|---|---|
| candidate bank | (G,5,3,2) float32 | world XY commanded displacement[m]，dt=.1s，逐分量±.06m |
| predicted margins | (G,5) float64，每事件一份 | signed risk/success[m]；selector唯一的逐候选信息 |
| frozen buffers/scale | 两scalar[m] / scalar无量纲 | 来自S4.3，默认5.476/15.516mm、scale1 |
| rejected | (G,5) bool | 当前gate是否禁止此候选；不是actual risk label |
| chosen | (G,) int64 | 0=保留primary，1…4=alternate，−1=显式零命令fallback |
| executed commands | (G,3,2) float32 | 一次选择后执行完整H3；不重新采样、重新clip或解释动作 |
| actual trajectory | (G,4,2) float32 | 每方法独立reset，保存初态与3个执行后XY状态[m] |
| clock | (4,) float64 | [0,.1,.2,.3]s模拟时钟，不是wall-clock耗时 |
| initial / goal | (G,2) | initial float32；goal probe float32 / VLA float64，world XY[m] |
| actual risk/success | (G,5) bool | bank逐候选真值，仅用于事后rejection统计，不传selector |

G=32为两个probe条件，G=16为两个VLA条件；每组5候选相关。旧VLA仍只有8布局×2指令，
probe是旧S4.3合成诊断数据，不是VLA输出，也不是新盲测集。
输出`contract.json` schema `chunk_rejection_alternate/1.0`、`decisions.npz` raw数据/选择/执行轨迹、
`report.json`全计数/来源/环境/latency、`summary.csv`方法对照；NPZ禁用pickle。

## Core Concepts

- 默认只按risk flag拒绝，不把未报success当作必须停止的理由；risk-free候选可以仍未到goal。
- `--require-success`显式增加success要求；保留S4.3全负退化，不为通过gate重调buffer。
- `select`不接truth/actual label/gain/goal坐标。goal信息已包含在预先算好的predicted success margin里。
- 替代动作仍来自原5候选；没有oracle controller、新规划器或重新训练。
- fallback是三条真实零命令，不是T=0拒绝记录，也不是把−1误当NumPy最后一条候选。
- 原18次proposal输入的2个unknown拒绝原样保留，**不会**进入gate或被伪造为零命令episode。
- 真正good定义为actual risk-free且h3 goal success；bad仅指actual risk事件。
  risk-free但goal失败的候选既非bad也非good，避免把“安全”偷换成“有用”。

## Mathematics

沿用S4.3 R̂、Ŝ和frozen B_risk/B_success，α来自source report。
默认reject_ik=[R̂_ik≥−αB_risk]。
严格模式reject_ik=[R̂_ik≥−αB_risk] OR [Ŝ_ik<αB_success]。
E_i={k | reject_ik=False}。

选择j_i：

- 若0∈E_i，则j_i=0。
- 若0∉E_i但E_i非空，选argmax_(k∈E_i) Ŝ_ik；并列取较小index。
- 若E_i为空，j_i=−1，A_exec=0_(3×2)。

因为Ŝ=.01−predicted final L∞ goal distance，argmax等价于在允许候选中选末点距离最小者。
这不是success probability最大化，也不承诺选中的候选会成功。

Bad_ik=[actual R_ik≥0]；Good_ik=[actual R_ik<0 AND actual S_ik≥0]。
Rejected_bad=Σ reject·Bad；false_rejected_good=Σ reject·Good。
另报rejected_risk_free=Σ reject·(NOT Bad)，其范围比false_rejected_good更大。
全部候选统计分母G×5；primary_rejection只看k0，分母G；三种执行方法各有G次attempt，包含fallback。

每方法独立plant执行p_(h+1)=p_h+gain_i·a_h，h0…2。
Gain只传evaluator，不传selector。每方法都从同一measured initial开始，无真值反馈重规划。
Failure stage按实际结果分成unsafe优先、risk_free_goal_miss，余下为risk_free_goal_success；三者合计G。
Goal success独立另报，可与unsafe重合，不能把两者计数相加当分母。

## Architecture / Data Flow

冻结S4.3 margins/buffers→gate→保留primary / eligible alternate / zero fallback→actual H3执行。
独立evaluator读取bank truth统计拒绝质量，再读取执行后的actual trajectory算risk/goal结果；
这些数据不进入selector。源文件只读，model readonly，检查前后hash。
对照三方法：`candidate0`总执行k0，`reject_only`拒绝后直接zero，`alternate`允许备选后才zero。

## Math / Architecture → Code

`load_scores`校验上游并整理四条件；`select`实现gate、primary优先和稳定argmax。
`selected_commands`只对index≥0显式gather，其余保留全零，防止−1索引误用。
`rejection_counts`分开bad/risk-free/good及primary子集；`replay`仅供独立实际执行。
`main`保存逐步轨迹、分母/失败stage、gate计时与source/code hashes，不导入或修改旧仓库。

## Minimal Experiment

Repo .venv CPU NumPy，无新依赖、安装、下载、训练或重新校准。
两条都使用相同S4.3 source。先读Problem→Mathematics，第二条先预测strict变化：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_4_rejection_alternate.py --output tmp/my_s4_4
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_4_rejection_alternate.py --require-success --output tmp/my_s4_4_strict
```

自有source两条均加`--scores tmp/my_s4_3`；下面实测使用margin-scale1的S4.3 source。
若source是α0产物，严格模式不一定全fallback，必须根据自己的source阈值解释，不能套用下述数字。
查看contract.json、report.json、decisions.npz、summary.csv，尤其核对−1→zero、fallback风险和good分母。

## Expected Result

Risk-only gate应复用S4.3标记，不改scores/labels/buffers/模型。
替代可减少本bank的监测风险，但不保证到goal或对未知机制安全。
α1严格模式因15.516mm success buffer大于10mm分数上界，所有候选均被拒，全部fallback。
严格模式虽然拒绝所有bad，也会误拒所有actual good；不能据此说gate更好。

## Actual Result

2026-10-20，repo .venv Python3.12.3 / NumPy2.5.4 CPU，实际模块在repo .venv。
默认source为`tmp/s4_3_final_checked`，α1；无训练/fit/下载。

### 候选拒绝质量

下列全部是bank计数，不是实际执行次数；risk-free被拒不一定等于good被误拒。

| 条件 | G×K | actual bad | 拒绝bad | actual risk-free | 拒绝risk-free | actual good | 误拒good |
|---|---:|---:|---:|---:|---:|---:|---:|
| probe ID | 160 | 24 | 24 | 136 | 6 | 26 | 2 |
| probe gain.65 | 160 | 23 | 23 | 137 | 7 | 30 | 0 |
| VLA nominal | 80 | 56 | 56 | 24 | 0 | 0 | 0 |
| VLA shifted gain.65 | 80 | 48 | 48 | 32 | 8 | 0 | 0 |

Strict拒绝所有候选：上述bad全拒，actual good也全拒（probe26/26与30/30）。
VLA actual good为0，误拒good计数0不能解释为完美保留有用动作。

### 一次H3实际执行

每格“unsafe / risk-free goal success”，分母为该行attempts；goal success独立计数见report。
本次这些方法的goal successes恰好都risk-free，但定义仍独立。

| 条件 | attempts/方法 | candidate0 | reject_only | alternate | alternate的primary/备选/fallback |
|---|---:|---:|---:|---:|---:|
| probe ID | 32 | 5 / 1 | 3 / 1 | 3 / 1 | 27 / 1 / 4 |
| probe gain.65 | 32 | 4 / 4 | 3 / 4 | 3 / 4 | 27 / 1 / 4 |
| VLA nominal | 16 | 16 / 0 | 0 / 0 | 0 / 0 | 0 / 16 / 0 |
| VLA gain.65 | 16 | 16 / 0 | 0 / 0 | 0 / 0 | 0 / 16 / 0 |

两个probe条件均有3个fallback起点在圆盘内，zero之后仍处于圆盘，fallback_unsafe=3。
虽然本bank allowed_bad=0，最终执行仍有unsafe fallback，说明拦截统计不能替代执行评价。
VLA primary被替换后监测risk从16/16降至0/16，但goal success始终0/16，没有任务成功改善证据。

Strict下`reject_only`/`alternate`在每个probe条件32/32 fallback、每个VLA条件16/16 fallback；
probe unsafe仍3/32且goal success0，VLA unsafe0/16且goal success0。
Baseline candidate0与非strict逐值相同；strict不改变模型、source、原动作/score/labels/gain。

Gate-only latency（每次全context批处理，30测量/5warmup）：
probe ID median/P95 .0221/.0231ms；VLA nominal .0645/.0787ms。
包含flags与candidate选择，不含WM、几何score计算、gather/plant/IO/真实sensor/controller延迟。
两个batch大小/分支数不同，不作公平速度排名或实时保证。
7项新增测试、全52项tests通过；独立重算selection/物理转移/风险与goal/所有分母、strict不变量通过。
NPZ round-trip与源hash一致；详见handoff。没有新闭环控制实验，report closed_loop_success=null。

## Result / Explanation

建立了可审计的拒绝→替代→fallback执行链，并分开评价gate质量和实际结果。
本次risk-only替代降低固定VLA chunk的监测风险，未让它到达目标；严格success gate则退化为不行动。
不能把probe结果说成learned VLA成功，也不能把open-loop chunk replay说成closed-loop benchmark。

## Failure Analysis / Limitations

| 情况 | 影响与证据边界 |
|---|---|
| zero停在禁入区 | 本probe有3例，fallback不具备安全保证；不隐式加入escape oracle |
| strict全拒 | 保留S4.3校准退化，拒绝全部good；未用评价真值调阈值 |
| risk-free不等于good | 未到goal仍可能通过gate；三类计数分开 |
| 选择偏差/机制变化 | bank gate覆盖不保证所选/fallback结果，缺真实概率或OOD保证 |
| 重复旧probe/layout | 不是新盲测，候选相关，不能当独立160/80场景 |
| 采样点risk | 不检查初态/连续线段/真实碰撞；不能称机器人安全控制器 |
| 理想goal grounding | 仍用S4.3当前RGB/literal解析假设，无新的语言视觉能力证据 |
| one decision/H3 | 没有prefix feedback或replan；不证明长时闭环或整机实时性 |

## Relation to Robotics

此层位于proposal verification与执行之间；真实系统的fallback需要结合当前状态、动力学和控制器设计。
本节明确保留zero失败，未修改plant/action语义或sibling repo。

## Interview Capsule / Notes

30秒：我把冻结risk flag接成gate，原动作通过则保留，否则从允许bank选最近goal的候选，无候选输出zero。
统计拦下bad和误拒good，再独立执行；VLA chunk风险降低但任务仍0/16，probe里zero还有3例停在禁入区。

2分钟：解释risk-free与good、全bank与primary/实际执行分母、稳定argmax与−1 sentinel，
strict success为什么全部fallback，以及为什么拦截bad不能证明fallback安全或闭环成功。

## Likely Follow-up Questions

1. 默认/strict gate公式与候选选择顺序是什么？candidate0通过时为何不换成最高分者，tie/无候选怎样处理？
2. selector输入哪些信息，truth/gain在哪里使用？bank/score/index/command/trajectory的shape、dtype、单位和时间是什么？
3. bad、risk-free、good如何定义？160/80候选、32/16执行attempts、primary rejection是什么不同分母？
4. 为什么allowed_bad=0仍可能执行unsafe fallback？本实验3个zero失败与VLA 0/16任务结果说明什么？
5. 打开require-success哪些量不变、哪些会变？为什么α1全fallback，误拒good怎样变化，为什么不代表更安全/更会控制？

## Must Remember

拒绝bad≠执行安全；risk-free≠good；zero≠安全；−1必须显式fallback；一次H3选择≠闭环重规划。

## My Verification

- [ ] Run：检查gate、选择index、实际commands/trajectory、bad与good分母以及fallback unsafe。
- [ ] Modify：先预测开启require-success的原数据不变量、全拒/全fallback与good误拒，再运行新目录核对。
- [ ] Explain：独立回答上述五问，区分候选统计、实际chunk结果与closed-loop success。

Engineering Complete；Learning未确认。Current Task=S4.4；Next Task=S4.5 — Receding-horizon Replanning仅计划，STOP。
