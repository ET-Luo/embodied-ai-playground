# S4.5 — Receding-horizon Replanning

## Problem

把冻结VLA proposal→world model→S4.4 gate接成execute prefix→observe→replan循环。
对照真实状态反馈与仅沿模型belief重规划，检查E1/E3、扰动、失败分母与latency。
本节是单checkpoint反馈时序学习实验；不进入S4.6的最终VLA-only vs VLA+WM matched benchmark。

## Intuition

每次先看当前位置，重新提一份三步计划，只执行前E步，再看真实落点。
如果实际执行有未知gain或kick，只在模型自己预测的位置上重算计划并不能获得新信息。
反馈能修正下一次规划起点，却不保证错误的VLA会朝正确目标运动。

## Input / Output Contract

[script](../../scripts/s4_5_receding_horizon.py)，[tests](../../tests/test_s4_5_receding_horizon.py)。
默认`--gate tmp/s4_4_final`，沿来源链检查S4.4→S4.3→S4.2→S4.1→S1.5五级文件，
包括schema/plant、source/model/proposal/weights hashes、固定阈值与参数shape/dtype。
缺失即报错，不隐式训练、校准或下载。所有model/weights readonly、文件hash运行前后不变。

| 项 | shape / dtype | 模态、单位、frame、时间与流向 |
|---|---|---|
| 当前RGB | (32,32,3) uint8 | 静态合成正交camera，同布局重render，无agent marker/未来帧 |
| text | literal string | move to red/blue；unsupported组grab red/blue block真实拒绝 |
| observed position | (2,) float32 | decision tick的actual world XY[m]，仅observed_state使用 |
| belief position | (2,) float32 | 初态真实；之后从选中model prefix endpoint转float32，无新观测 |
| VLA proposal bank | (5,3,2) float32 | 原learned H3+固定bias候选，world XY command[m]；dt=.1s/逐分量±.06m |
| model predicted bank | (5,4,2) float64 | 当前planning state+三个AR未来world XY位置[m] |
| score / flags / chosen | (5,) float64[m] / (5,) bool / int | 原S4.3 margin、S4.4 gate/primary优先；−1显式zero |
| actual episode state | (T+1,2) float32 | 实际测量world XY[m]，含gain/kick效果，原物理数据保留 |
| executed action | (T,2) float32 | 真正发出的命令，不把外界kick改写成action |
| clock | (T+1,) float64 | relative simulated seconds，dt=.1s；与wall latency分开 |

正常episode预算T8，state(9,2)、action(8,2)；不支持语言T0，只有state(1,2)、action(0,2)，
保留拒绝attempt，无伪造zero轨迹。`executions.npz`各case单独key保存可变长度数组，allow_pickle=False。
`protocol.json`保存64cases与外扰声明；`report.json`含128条method×case记录、完整bank/prediction/选择/
实际prefix和计时；`contract.json` schema `vla_world_model_feedback/1.0`声明模块与结果语义。
不是把T0记录冒充embodied_episode/1.0有效episode。

## Core Concepts

- 冻结S1.5 VLA及S4.1 bias bank，每个decision都重新提案，无warm suffix、重训或额外oracle动作。
- world model与train-only statistics冻结，gate继承S4.4 source的buffers/scale/strict开关。
- 规划goal由当前RGB与literal解析，延续S4.3理想grounding假设；evaluator goal annotation只评分。
- observed_state每次规划刷新actual state；predicted_state只从t0真实起点沿model belief推进。
- 实际执行结果仅在observed_state的下一decision边界进入控制，不在prefix中途重算动作。
- **预测/gate始终H3**。执行min(E,8−t)条并丢弃旧后缀；最后可能验证预算外的未来，
  例如E1在t7预测到t10，却只执行到t8。这可能因未执行尾部而拒绝候选，明确保留此限制。
- Full-H3 success margin是rolling窗口末点分数，不是对episode t8终点成功的保证。
- E改变不修改H、K、dt、模型、阈值或protocol；不保证任务success或latency单调。
- 冻结buffer原在手工probe上校准，并未对本轮新visited states再次校准；没有风险覆盖保证。

## Mathematics

B_t=FrozenVLA(RGB_t,ℓ,planning_state_t)→5个H3候选。
P̂_t=FrozenWM(planning_state_t,B_t)；gate按原margins选择chunk C_t或zero。
执行n_t=min(E,8−t)：p_(t+h+1)=p_(t+h)+g·C_t[h]+k_(t+h+1)。

名义g1/k0；gain条件g.65；kick条件g1且k₂=[.12,0]m，其余k为0。
Gain和kick只在plant/evaluator内，VLA/model/gate不获知外扰参数。

下一次planning state：

- observed_state：p_actual_(t+n_t)。
- predicted_state：float32(P̂_selected[n_t])，fallback也由同一冻结模型预测zero chunk后更新belief。

E1 decision ticks0…7，prefix均1；E3 ticks0/3/6，prefix3/3/2。
Kick在第2次transition后发生：E1下次t2读取新状态；E3第三条命令仍来自t0计划，到t3才刷新。
所有case执行8步，不因提前经过goal或风险而提前停止；success始终看最终t8并检查已执行风险。

Episode goal_success=[completed8且||p₈−g_eval||∞≤.01m]。
Episode unsafe=[任一已执行p₁…p₈在监测圆盘内]，不检查初态、连续segment或真实contact。
Success=goal_success AND NOT unsafe AND 无runtime拒绝。
失败stage：policy_input_or_output_rejection（原proposal ValueError）、monitoring_risk、final_goal_error_after_budget。
Stage是pipeline症状，不是对视觉/语言根因的结论。

执行前缀prediction RMSE：只比较每decision实际执行的n_t个位置与该decision预测的对应位置，
Σe² / (2Σn_t)后开方，单位m。每完整16case组分母16×8×2=256；全拒绝分母0→null。
未执行H3后缀不加入误差；predicted_state输入是belief，误差含此前漂移，不能视为共同真state上的单步精度。
跨E也改变lead time与visited states，不能把误差差异全归因为模型本身或做公平模型排名。

## Architecture / Data Flow

当前RGB/text/observed-or-belief state → frozen VLA+biases → frozen WM H3 → margins/gate
→ chosen prefix → actual gain/kick plant → 原始next state。
Observed分支把下一decision实测state回送；predicted分支只回送model prefix endpoint。
独立evaluator读实际轨迹做goal/risk/预测误差；不把评分标注传回控制器。

## Math / Architecture → Code

`load_system`沿S4.4来源链加载冻结参数/阈值；`protocol`固定四条件、8layout×2texts及外扰。
`run_episode`在while t<8里重提案/预测/选择，逐条执行前缀，再更新下一decision起点。
`summarize`保留全attempt分母、零动作拒绝、failure stage、真实action scalar误差与分阶段计时。
`main`做五次完整episode warmup并排除，保存所有source/code hashes、128attempt记录和NPZ逐字段round-trip。

## Minimal Experiment

CPU repo .venv / NumPy，无新依赖/安装/下载/训练/重新校准。
先读Problem→Mathematics，第二条先预测E1→3的调用时刻、末前缀、kick反馈及compute变化：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_5_receding_horizon.py --output tmp/my_s4_5
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_5_receding_horizon.py --execute 3 --output tmp/my_s4_5_e3
```

自有S4.4 source，两条均加同一`--gate tmp/my_s4_4`。
下面实测source是risk-only、scale1；若输入strict或α0产物，不能套用此结果。
查看contract.json、protocol.json、executions.npz和report.json，核对planning_state与actual/belief时间边界。

## Expected Result

E3调用8→3、末prefix2，first decision bank应与E1相同；来源/参数/阈值/protocol均不变。
名义条件两分支几乎重合；gain/kick下真实反馈能刷新规划起点，belief分支不能消除外扰偏差。
原VLA仍可能失败，不能由replan调用次数或低局部预测误差推断任务能力。

## Actual Result

2026-10-20，repo .venv Python3.12.3 / NumPy2.5.4 CPU，实际模块在本repo .venv。
每run4条件×16cases×2methods=128attempts，来自旧8布局，条件共享场景，不是128独立环境。
两个run所有goal_success/success均0/16，已执行监测unsafe均0/16；
前三组各完成16条8步，stage全部final_goal_error_after_budget；unsupported各16条T0拒绝，
stage为policy_input_or_output_rejection，prediction RMSE=null，正常accepted latency=null。

| 执行前缀RMSE mm | observed E1 | belief E1 | observed E3 | belief E3 |
|---|---:|---:|---:|---:|
| nominal | .000002794 | .000006202 | .000002678 | .000004261 |
| gain.65 | 12.354 | 43.198 | 14.478 | 30.785 |
| kick | 30.000 | 79.373 | 42.426 | 79.373 |

Kick解析检查：E1刷新前只有第2个state带120mm未知x偏差，整episode误差=120/√16=30mm；
E3第2、3个state带偏差，120√(2/16)=42.426mm；belief分支第2…8共7个state都保留偏差，
120√(7/16)=79.373mm。这解释时序效应，不代表模型学会了扰动机制。

完整known case组每方法每条件E1有128accepted plans，E3有48；单episode8→3，prefix3/3/2。
本次没有fallback plan，未删除fallback代码；unit test用strict门控验证zero仍执行/再次重规划。
Unknown每method保留16rejected attempts；5次完整warmup episodes不进统计。

### Latency

每accepted decision分别测render/proposal/world_model/score_gate、planning总和及plant apply。
score_gate含当前goal解析、margins、选择与命令gather，若fallback还含zero模型预测；
planning包括前四项，不再和它们相加。apply仅本地toy更新；不含真实sensor/通信/servo等待或IO。
Rejected attempt单独计时，不混入正常accepted median/P95。

Nominal observed E1 planning median/P95=.2525/.3550ms，128calls；每episode规划计算median2.0754ms。
E3为.2673/.4577ms，48calls；每episodemedian.8056ms。单次耗时不保证随E单调，calls减少能减少本次总compute。
计算预算为n_t×100ms，未观察到overrun；这不是异步deadline或整机实时保证，也未实际sleep模拟时钟。

7项新增测试与全59项通过；独立审计两run共256attempt的物理gain/kick、prefix、T/T+1/dtype/clock、
feedback/belief起点、no-feedback跨条件commands不变、跨E首decision/source不变、误差scalar与全部failure分母通过。
输出source hash、NPZ round-trip、依赖/文档检查与准确命令见handoff。

## Result / Explanation

工程完成了真实状态刷新后的VLA→WM→gate重规划链路；同一来源的belief对照证明重算本身不等于反馈。
真实反馈降低本次机制偏移下的预测偏差，但没有修复失败策略，goal仍0/16。
这是ideal静态RGB、精确proprio的toy反馈控制，不是实际机器人或最终benchmark结果。

## Failure Analysis / Limitations

| 情况 | 影响/当前证据 |
|---|---|
| VLA仍失败 | 所有16case组终点未到goal；不接oracle修复或扩大训练 scope |
| belief重算无实测 | gain/kick偏差累积，调用更多次不等于获得新信息 |
| H3末尾超预算 | gate可能因实际不执行的尾部拒绝；rolling success不是t8保证 |
| stale prefix | E3在kick后仍执行旧slot，t3才刷新；必须记录外扰与decision时序 |
| 旧probe校准 | buffers未覆盖新visited分布，不能用本次unsafe0推断风险保证 |
| 0动作拒绝 | 分母保留、RMSE null，不能伪造8条zero命令或删除这些attempt |
| fallback | 本次未触发；不能据此忘记S4.4 zero可停在禁入区的失败 |
| 理想观察/goal | exact state与完美RGB goal decoder，没有camera delay/noise/learned grounding改进 |
| reused layouts/single model | 非新盲测、无独立种子稳健性或OOD泛化声明 |
| timing | 本地compute不是完整机器人链路，也不是最终系统benchmark |

## Relation to Robotics

让policy/world model输出通过明确动作契约进入feedback执行环；真实系统还需状态估计、控制器、
时钟同步和连续风险监测。原physical command、external kick与measured delta保持分离；旧仓库未改动或导入。

## Interview Capsule / Notes

30秒：冻结VLA与WM，每轮预测H3、gate后执行E步，再刷新实测state；belief-only对照不读新state。
E1/E3的kick观察时刻不同，前缀RMSE30/42.43mm；belief约79.37mm，但策略goal仍0/16。

2分钟：解释E/H/remaining budget、丢弃旧后缀、固定H3 gate超预算限制、输入拒绝全分母、
为何反馈修正起点不等于策略修复，以及局部compute与整机实时/最终benchmark的区别。

## Likely Follow-up Questions

1. 每轮数据流与shape/dtype/单位是什么？observed和belief的下一次planning state分别来自哪里，哪些信息只给evaluator？
2. E1→3的ticks/prefix/kick观察时刻怎样变化？最后剩2步时H3 gate看了哪些未执行未来，怎样丢弃旧后缀？
3. 为什么belief分支在nominal/gain/kick下的commands保持相同？推导kick下30/42.426/79.373mm误差与256 scalar分母。
4. 128attempt、16case组、accepted plans与T0拒绝各是什么分母？unsafe0却success0意味着什么，stage/null如何解释？
5. E改变哪些量必须不变？如何解释分阶段/每episode latency、overrun0，以及旧buffer/理想观察/旧布局的证据边界？

## Must Remember

重算≠反馈；固定H3与执行E分开；记录外扰时刻；只评价已执行prefix；反馈减少偏差≠策略成功。

## My Verification

- [ ] Run：检查循环中的planning state、selected prefix、实际next state、failure与拒绝记录。
- [ ] Modify：先预测E1→3不变量、ticks0/3/6、末prefix2、kick误差与compute变化，再新目录运行。
- [ ] Explain：独立回答上述五问，区分反馈时序证据与任务成功/最终benchmark。

Engineering Complete；Learning未确认。Current Task=S4.5；Next Task=S4.6 — Final Benchmark仅计划，STOP。
