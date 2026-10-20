# S3.7 — MPC-style Replanning

## Problem / Why

S3.6一次规划执行10步，未知gain下只有6/20成功。本任务固定同一个checkpoint，
比较一次执行、只重复搜索和真实状态反馈重规划，检验observe→plan→execute prefix→observe的作用。
这是有限任务、剩余时域缩短的MPC-style shooting，不实现无限时域最优控制或稳定性证明。
预计1.5小时：概念/数学25分钟，代码20分钟，Run/Modify20分钟，Explain/失败分析25分钟。
只做S3.7，不开始video world model或Stage4集成。

## Intuition

按旧地图走完一整段，执行偏差可能一直累积。每走几步重新确定当前位置，再规划剩余路程，
有机会纠正偏差；但仅多算几次、仍沿用自己想象的位置，不等于获得了环境反馈。
更频繁观测/规划会增加计算，也不保证在所有模型、候选、时延条件下更好。

## Input / Output Contract

profile `s3_7_wall_feedback/1.0`。读取并冻结
[S3.6](s3_6_model_based_planning.md) `s3_6_wall_planning/1.0` checkpoint，
真实执行仍是S3.5 wall toy；不重训、不估计gain、不接learned reward/risk head或latent。

| 项 | shape / dtype | frame/单位/时间 |
|---|---|---|
| source model | 226参数float64 + train统计 | S3.6 wall residual MLP；训练800/验证400不变 |
| initial state | (20,2) float32 | 原held-out reset world XY [m]，全部保留 |
| actual states | 每condition/method (20,11,2) float32 | 10次实际转移、11个观测；world XY[m] |
| executed commands | (20,10,2) float32 | world XY commanded displacement[m]，dt=.1s |
| plan bank at t | (N,10−t,2) float32 | 默认N128，含真实零命令和未执行旧计划后缀 |
| model prediction | (N,11−t,2) float64 | 从此次planning state递归的剩余未来[m] |
| planning state | (2,) float32或float64 | feedback用实际XY；无反馈对照用模型belief |
| per-call cost | (N,) float64 | S3.6无量纲task cost应用到剩余段 |
| timestamp | (11,) float64 | 相对秒，0…1s，无terminal action |

只有理想完整XY观测，没有图像、语言、速度、力、噪声、估计器或传感延迟。
归一化统计原样复用S3.6 train800；更新状态不等于重fit normalizer或更新模型。
loader检查source schema/plant profile、模型shape/finite/正尺度、source轨迹wall step对齐、
episode partition、初态与train-only统计；模型hash与S3.6 report一致，全部source文件运行前后hash不变。

默认source `tmp/s3_6_final`。缺少source明确失败，不隐式重训或下载。
`--checkpoint tmp/my_s3_6`可使用自己已完成的S3.6目录；匹配的模型可不同，但不能把本包固定结果推广给不同checkpoint。

## Core Concepts

三种方法共享模型、初态、任务cost和candidate生成规则：

| 方法 | 下一次planning state | prefix | 含义 |
|---|---|---|---|
| open_loop | 只有t0真值 | 固定10 | 与S3.6 learned一致，执行整段 |
| replan_no_feedback | 上一计划预测的prefix终点 | CLI E | 多次搜索，但t0之后不接收真实状态 |
| mpc_feedback | 已执行prefix后的真实当前XY | CLI E | 闭环状态反馈，每次修正规划起点 |

no-feedback仍在真实plant执行动作，但真实位置只用于plant演化/评价，不进入后续规划选择。
feedback不知plant gain或扰动安排，只能在观测边界看到已经发生的结果。
E10下三方法只有t0一次规划，全部轨迹应相等，是关键退化检查。
这里可以称为**synthetic toy状态反馈闭环**，不能称为真实机器人闭环验证。

## Mathematics

总任务T=10，dt=.1s，planning times按实际执行前缀推进：

\[
R_t=10-t,\quad L_t=\min(E,R_t),\quad
\hat s_{t|t}=s_t\quad\text{(feedback)},
\]
\[
\hat S^{(i)}_t=\operatorname{Rollout}(\hat f_\theta,s_t,A^{(i)}_t),\quad
 i_t^*=\arg\min_iJ_{R_t}(\hat S^{(i)}_t,A^{(i)}_t).
\]

仅执行选中序列的前L_t步，然后t←t+L_t，用真实s_t重新初始化。
no-feedback把上式的真实s_t替换为上一计划的预测prefix终点；模型θ与全部统计始终冻结。
剩余预测始终只递归模型，没有真实未来插入。

E1：t=0…9，各R=10…1，每次执行1步，共10次规划。
E4：t=0,4,8，R=10,6,2，执行4,4,2步，共3次；最后不能越过任务终点。
E10：t0一次规划，共10步执行。所有方法都执行恰好10次转移，不因到达goal提前停或删失败。

### Candidate / warm start

调用S3.6 `candidate_bank(episode_id+1000*t,N,10−t)`，因此候选种子取决于原episode和绝对t，
t0与S3.6完全一致。后续plan的最后一个candidate替换成上一计划**尚未执行的后缀**，
总数仍N，index0零动作保留。旧动作不会从头重复，也不把padding当执行命令。

相同t的基础随机bank规则一致，但feedback改变历史决策后，warm-start后缀可能不同；
不能声称所有方法后续完整bank逐值相同。比较匹配的是proposal规则、N和状态反馈消融。

### Cost与任务边界

复用S3.6 `costs`：终点距离、途中平均距离、未来越界计数、平均命令能量。
每次在R_t剩余段上计算，path/effort均值的分母随R_t变化；
不是原始整段目标的严格Bellman分解，没有稳定性/最优性保证。
最终actual cost统一对完整10步已执行轨迹计算，便于跨方法对照。

成功规则沿用S3.6：末状态goal距离≤.045m且10个未来state无指定边界违规。
失败先记boundary_violation，否则goal_not_reached，记录首次违规转移；t0不进入违规历史。
完整20初态，所有失败保留，不把预测成功或在线cost下降当实际success。

### Perturbations与时序

1. nominal：S3.5 wall step。
2. weak_gain：plant内部将命令乘.65再step；模型仍以原命令预测，不向planner透露gain。
3. kick：nominal plant在绝对transition4→5执行后加world y位移+.08m，仅一次。
   E1在下一次t5规划观察到结果；E4在t4规划时尚看不到，先执行至t8再观测。

扰动不改变命令frame或单位，执行raw state包含真实偏移。没有未来kick预告、在线gain辨识或模型自适应。

## Architecture / Data Flow

`frozen S3.6 model + statistics + held-out starts → observation / belief at t`

`→ remaining candidate bank + old unexecuted suffix → predicted future → known cost → choose`

`→ execute L_t commands in actual plant → retain raw states/actions`

`→ feedback: actual prefix end / ablation: predicted prefix end → next planning call`

`→ actual full-trajectory success / failure / cost + planning count / latency`

source未来轨迹只用于loader/离线证据检查，不供controller规划；真实plant只在选择之后执行。

## Math / Architecture → Code / Implementation

[脚本](../../scripts/s3_7_mpc_replanning.py)：`load_source`做只读契约/统计/hash审计；
`planning_bank`生成剩余段并正确接旧后缀；`run_episode`在观测边界更新起点，
`plant_for`将gain或kick留在plant侧；最终`outcome/costs`复用S3.6评价规则。
controller获得plant callable而不读取condition/gain，评价层负责指定执行条件。

`executions.npz`保存20个初态、source IDs、时钟和九组(condition×method)完整raw state/action。
`report.json`保存180条episode记录、每次plan的起点/候选cost/选择索引/所选完整剩余动作和预测、
prefix端点误差、阶段计时；contract.json声明全部语义。
全部bank可由episode/t/N与上一计划后缀重建，无object NPZ、无padding歧义；读取禁用pickle。

[测试](../../tests/test_s3_7_mpc_replanning.py)：E4的4/4/2与真实起点、无反馈动作不受plant变化影响、
E10退化为单次规划、warm-start只包含未执行后缀。Stage3此前测试同时回归。

## Minimal Experiment

repo .venv Python3.12.3 / NumPy2.5.4，CPU，无训练、新依赖、下载或旧repo访问。
先准备已完成的S3.6 source；已有默认source可直接运行，自己的路径用`--checkpoint`指定。
输出目录必须不存在：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_7_mpc_replanning.py --output tmp/my_s3_7
# 先预测E1→4的调用时刻/shape、最后前缀、模型与open-loop不变量及计算/反馈取舍：
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_7_mpc_replanning.py --execute 4 --output tmp/my_s3_7_e4
# 可选退化核验：三方法应完全等价
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_7_mpc_replanning.py --execute 10 --output tmp/my_s3_7_e10
```

读取report summary时先比较相同条件下feedback/no-feedback，而非只比open-loop。
重复规划有更多candidate搜索和CPU预算；no-feedback对照帮助识别反馈作用，但单seed/有限bank不构成一般因果或鲁棒性定理。

## Expected Result

source hash/模型/统计/离线误差和初态保持不变；各E的open-loop应逐值相等并复现S3.6 learned。
无反馈方法的动作跨nominal/weak_gain/kick保持相同，feedback则可能改变后续动作。
E增大通常降低规划次数，但不保证更差或更好；cost、候选采样、模型误差都影响实际结果。
E10三方法应等价。MPC会修正起点，却不自动修复动力学、保证约束或识别执行器gain。

## Actual Result

2026-10-20，冻结默认S3.6模型，离线RMSE仍.373134mm，N128/T10：

| 条件 | open-loop | replan无新状态 E1 | 真实反馈 E1 | replan无新状态 E4 | 真实反馈 E4 |
|---|---:|---:|---:|---:|---:|
| nominal成功/20 | 19 | 20 | 20 | 20 | 20 |
| weak_gain成功/20 | 6 | 3 | 19 | 3 | 18 |
| kick成功/20 | 2 | 0 | 20 | 0 | 20 |

本轮所有失败均为goal_not_reached，boundary_violation均0；逐episode与失败分母全部保存。
E1 feedback的weak_gain仍1/20失败，E4仍2/20失败，没有通过调参隐藏这些case。
E10三方法分别复现nominal19/20、weak_gain6/20、kick2/20，完整执行轨迹逐值一致。
nominal/weak_gain open-loop与S3.6 source候选实际执行轨迹逐值一致。

每episode调用数：open-loop1，E1为10，E4为3；三条件×20ep的feedback总调用数分别600/180。
E1 feedback nominal/weak/kick平均final distance约2.910/11.778/2.846mm；
E4为4.826/17.890/11.532mm。成功率相同也不代表终点精度或过程cost相同。
nominal E1 no-feedback20/20也优于open-loop19/20，提示重复搜索本身就可能改变结果；
weak/kick下no-feedback明显失败，更支持当前toy中真实反馈有价值。

### 实测计算开销

以下包含候选生成+rollout+评分+选择，两项逐调用实测后相加再统计，无warmup：

| feedback条件 | E1每调用 p50/p95 [ms] | E1每episode总量 p50/p95 [ms] | E4每episode总量 p50/p95 [ms] |
|---|---|---|---|
| nominal | 2.072 / 5.132 | 21.604 / 39.613 | 7.626 / 11.655 |
| weak_gain | 2.356 / 4.032 | 23.404 / 31.837 | 6.932 / 9.323 |
| kick | 2.321 / 4.098 | 23.841 / 34.048 | 7.381 / 10.478 |

每调用100ms×实际prefix长度的计算预算对照overrun均0。
这不是实时机器人deadline实验：plant在compute期间不会继续演化，未模拟sensor/I/O/调度延迟。
plant step计时另存report，不包含在上表，真实1秒任务时长也不是脚本wall time。
27项Stage3测试通过（此前23项+本轮4项），并逐次重建bank/cost/argmin、重放全部plant轨迹核验。

## Explanation / Result

当前固定toy中，真实观测反馈能显著纠正未知gain和一次位移扰动的影响；
并不代表识别了新动力学，模型本身仍错误。无新状态的重复规划不能替代反馈。
实验是完整状态无噪声的synthetic closed loop，不是视觉policy、真实机器人或稳定性证明。
E1/E4取舍仅属这组候选/时域/算力，不保证更频繁规划普遍更好。

## Failure Analysis / Failure Cases

| 问题 | 当前处理 / 限制 |
|---|---|
| 模型重训使比较失去意义 | 冻结source model/statistics，hash前后相同，无隐式训练 |
| 重复搜索误称反馈 | no-feedback使用预测belief，跨plant动作完全不变 |
| 前缀执行后仍从旧s0规划 | 每次记录planning_state，feedback逐值等于已执行state[t] |
| 旧动作从头执行/最后越界 | warm只接未执行后缀；E4最后执行2步，T始终10 |
| 新bank数量混淆 | last candidate替换warm，总数仍N；方法间后续warm可不同 |
| 未来扰动信息泄漏 | plant才处理gain/kick，planner只看到已经发生的当前状态 |
| 把remaining cost视为全局最优 | 平均项随remaining长度变化，非严格Bellman一致或稳定性证明 |
| 规划多就算更公平 | 明报调用数/总compute，no-feedback对照仍非相同预算open-loop |
| 成功后提前停导致分母不同 | 固定10步不早停，全部20初态/所有失败保留 |
| zero overrun当实时控制证明 | 瞬时toy plant无调度/通信/传感延迟，未测实际deadline |

## Relation to Robotics

receding-horizon控制将预测与反馈接起来，但仍依赖观测质量、模型适用范围、候选覆盖和计算周期。
真实系统的延迟、隐藏速度/接触与约束需要额外实证，不能从这里的理想XY反馈直接推广。
本任务止于S3.7，不进入S3.8大型视频world model概念包或Stage4。

## Interview Capsule / Interview Notes

30秒：我冻结S3.6动力学，比较单次规划、预测belief重规划和真实XY反馈重规划。
同20个held-out初态、128候选、10步任务，未知gain下成功分别6/20、3/20、19/20。
E4把规划次数10降到3，反馈成功18/20；模型未适应gain，只是每次重置真实起点。
这是synthetic闭环证据，未证明真实机器人鲁棒或实时安全。

2分钟：补充remaining horizon/执行前缀区别、E4的4/4/2、warm后缀、无反馈消融、
完整失败分母、source freeze与open-loop复现，以及每次/总compute和模拟时延边界。

## Likely Follow-up Questions

1. open_loop、replan_no_feedback、mpc_feedback的planning state来自哪里？为何反复调用planner不等于获得反馈？
2. E1/E4/E10的调用时刻、remaining horizon和实际前缀长度分别是什么？warm start如何对齐，哪些shape/物理语义不变？
3. 为什么冻结模型仍可改善unknown gain结果？为何这不叫模型适应、gain识别或稳定性保证？
4. E1→4哪些artifact/open-loop结果不变，哪些计算预算、观测时刻与后续候选会变？是否保证实际success单调变化？
5. 如何核验没有未来泄漏、旧起点或错误索引？怎样解释no-feedback对照、全分母失败与latency，为什么不是真实机器人实时闭环证明？

## Must Remember

重新搜索≠新观测；反馈修正起点≠修复模型；remaining horizon≠执行前缀；冻结比较与全分母；toy实时预算≠真实deadline保证。

## My Verification / Learner Verification

- [x] Run：运行E1，核对三方法/三条件及source hash、完整失败与计算开销。
- [x] Modify：先预测E1→4，再运行对照，检查t=0/4/8与最后2步；可选E10退化核验。
- [x] Explain：用自己的话回答五问，画观测/预测belief的不同反馈路径。

Engineering Complete + Learning Mastered（2026-10-20 用户明确确认实验和预测完成，并准确回答五项Explain）。
解释覆盖真实state与预测belief反馈路径、E1/4/10的时域/前缀/调用时刻、warm后缀、
冻结模型纠偏但非gain辨识/稳定性证明、E变化不变量与计算取舍、重放审计及toy闭环证据边界。
Current Task=S3.7；Next Task=S3.8仅计划。
STOP，不自动开始下一任务。
