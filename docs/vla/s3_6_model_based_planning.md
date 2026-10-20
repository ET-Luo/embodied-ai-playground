# S3.6 — Model-based Planning

## Problem / Why

已有动力学模型，但“预测得准”和“能选出好动作”是不同问题。
本包学习 `candidate action sequences → model rollout → task cost → argmin → plant execution`。
训练一个匹配S3.5墙toy的小动力学模型，在20个held-out reset上比较实际执行结果。
每次只在t0规划一次，整段10步执行完毕；不在本任务提前实现S3.7重规划或MPC。
预计1.5小时：概念/公式25分钟、代码20分钟、Run/Modify20分钟、失败分析/Explain25分钟。

## Intuition

先想象若干种走法，给每种预测的未来打分，选择最低代价的一条，再实际走完。
可能失败在三个地方：候选里根本没有好走法；模型把坏走法排在前面；执行机制与模型不一致。
增加候选只能帮助覆盖更多走法，不会自动修复错误模型。用模型自己预测的成功判断自己，不能验证控制结果。

## Input / Output Contract

新profile `s3_6_wall_planning/1.0`，环境语义沿用
[S3.5](s3_5_outcome_heads.md) `s3_5_wall_outcomes/1.0`；不能将S3.1无墙checkpoint直接用于墙环境。
本轮自包含重新拟合墙动力学，不改旧checkpoint；不接入S3.4 latent或S3.5 learned reward/risk heads。

| 项 | shape / dtype | frame、单位、时序 |
|---|---|---|
| 原训练/验证源state | (60,21,2) float32 | S3.5 world XY位置[m] |
| 原source action | (60,20,2) float32 | commanded world XY displacement[m]，dt=.1s |
| train输入 | (800,4) float32→float64 | 当前XY+命令XY，train-only标准化 |
| train目标 | (800,2) float32 | 实际next_state−state，按train统计标准化 |
| benchmark初态 | (20,2) float32 | 20条held-out source episode的t0，全部保留 |
| 候选动作bank | (20,N,10,2) float32 | 默认N128，每轴命令≤.06m，完整未来1秒 |
| model rollouts | (20,N,11,2) float64 | t0真实初态，其后递归预测world XY[m] |
| plant rollouts | 同上float64存储 | 环境每步计算float32，逐候选独立reset后保存 |
| costs | (20,N) float64 | 显式无量纲任务目标，越小越好 |
| selected index | 每attempt一个整数 | learned/ideal先选择，之后才由evaluator查实际未来 |

无image/language/force/velocity/contact sensor。goal、wall和边界固定，非任意任务/形状泛化。
源数据seed350、split351，train40episode/800转移，validation20/400，
输入与实际位移统计只fit train800，226参数4→32 tanh→2 residual MLP，seed360/2000步Adam。
MLP输入显式float32以保持原始normalization精度，网络与预测状态float64；不对预测状态偷偷施加真实墙clip。
模型必须从数据学习墙行为；仅ideal comparator和真实plant使用声明的墙规则。

benchmark horizon H=10（1s），与source训练episode的20步长度不同；不把terminal后拼接成新轨迹。
每次plan后执行**完整10步**，没有途中观测反馈、重新选择或根据环境结果改动作。
原始source、候选动作与所有预测/实际状态均保存，模型不接收未来source轨迹。

### 两个执行条件

- nominal：S3.5真实toy step，nominal attempted XY动力学后x截断到.12m。
- weak_gain：执行器在plant内部将命令乘.65，再执行同一wall step；候选存的仍是原始命令[m]。
  这是显式未建模的执行增益变化，不是重新定义action单位，也不向planner透露gain。

两条件使用相同初态/候选，learned与ideal选择索引完全相同，不重新训练或适应weak_gain。
这里“实际环境”指独立于learned predictor的**synthetic plant函数**，没有物理机器人或MuJoCo实验。

## Core Concepts

- Random shooting：采样有限候选，批量预测，按任务cost选最好者；不保证连续动作空间全局最优。
- 模型只预测转移；任务cost可显式给出，不要求所有cost都由learned reward head预测。
- Open-loop规划：只t0看真实状态，执行整段；MPC-style observe/replan留给S3.7。
- 候选覆盖与候选排序分开：即使真实环境oracle也无法从缺少成功轨迹的bank里选出成功轨迹。
- 模型误差可能被优化器利用；预测cost更小不保证实际cost更小。
- 本包成功是执行后的task outcome，不是“预测goal概率”或离线验证loss。

## Mathematics

### 动力学

\[
\hat s_{h+1}=\hat s_h+\mu_\Delta+\sigma_\Delta\odot
\operatorname{MLP}_\theta(([\hat s_h,a_h]-\mu_x)/\sigma_x),\quad \hat s_0=s_0.
\]

所有s、a是world XY米单位向量，dt=.1s。train statistics fit800条转移，Adam lr=.01、
β=.9/.999、ε=1e-8；复用S3.1梯度和S3.3显式训练循环，未引入新黑盒库。

### 候选

N个候选长度H10。index0恒为零动作；其余按candidate-specific seed
`360000+source_episode_id*1000+candidate_index`生成：
随机active duration 1…10、随机switch、第一段uniform[-.06,.06]m，
第二段加std .02m扰动并clip到命令范围，active结束后补真实零命令。
这是有意义的“停止执行”命令，不是padding；所有10个动作都实际执行。
候选生成不看真实未来，也不使用goal指导手写expert路径。
index1为固定random comparator，不从实际结果择优。
N改变时小bank严格是大bank前缀；模型训练数据/初始化与训练预算不变。

### 代价

设g=[.09,0]m，future steps h=1…10，
\(v(s)=\mathbf1[s_x<-.20m\lor |s_y|>.20m]\)：

\[
J(S,A)=\| (s_{10}-g)/.1m\|_2^2
+.05\frac1{10}\sum_{h=1}^{10}\|(s_h-g)/.1m\|_2^2
+5\sum_{h=1}^{10}v(s_h)
+.01\frac1{20}\sum_{h=0}^{9}\sum_{d=1}^{2}(a_{h,d}/.06m)^2.
\]

所有项无量纲；.1m/.06m是事先声明的评分尺度，不是从validation fit的normalizer。
边界项是每个违规未来时间槽一次惩罚，不是calibrated probability；contact本身不算failure。
代价是terminal距离、途中距离、越界次数与命令能量的取舍；它**不等同于最大化二值success**，
因此最低cost oracle也不是最大success oracle，更不是所有候选以外的全局最优。

\[
i^*=\arg\min_iJ(\hat S^{(i)},A^{(i)}).
\]

learned planner只访问预测轨迹。环境执行后才能得到 \(J(S_{plant}^{(i^*)},A^{(i^*)})\)。
为了诊断，evaluator可在toy独立reset下测试所有候选：
\(i_{oracle}=\arg\min_iJ(S_{plant}^{(i)},A^{(i)})\)，
regret=selected actual cost−oracle actual cost≥0，**只相对当前bank**。
真实系统不能无成本试完所有未来，oracle仅是特权评价参照。

### 成功与失败

成功：执行完10步后距离goal≤.045m，且10个未来状态均无边界违规。
只评价future states，不把t0加进成功规则；全部20个初态保留，包含少量初态已越界情况，可能首步返回边界内。
失败优先分`boundary_violation`（记录第一次违规transition index），否则`goal_not_reached`。
每method/condition分母20，不删失败。这个open-loop执行率不能称为MPC/机器人闭环成功率。

## Architecture / Data Flow

`S3.5 source episodes → whole-episode split → train-only statistics → fit wall dynamics → freeze`

`held-out initial state → fixed candidate bank → learned rollout → known task cost → select index`

`selected full chunk → independent actual plant → actual states / cost / success / failure stage`

`evaluator only: all candidates in plant → privileged oracle / coverage / candidate-set regret`

代码先固定learned/ideal索引，再访问该reset的任何真实候选未来。
真实结果不进入candidate seed、模型、评分权重或选择分支。

## Math / Architecture → Code / Implementation

[脚本](../../scripts/s3_6_model_based_planning.py)：`candidate_bank`实现嵌套候选，
`predict_state`恢复物理delta，`rollout`显式循环未来10步、并行candidate轴；
`costs`实现公式，`select`仅调用提供的predictor，`outcome`计算实际成功/失败。
NaN预测/cost明确拒绝，不用`argmin`默默选择无效轨迹。

比较五个方法：learned、ideal（已知墙clip但假设s_next=s+a）、zero(index0)、random(index1)、
oracle（相同cost/candidates的真实plant rollout最优）。ideal不是实际真实动力学。
所有方法共享初态、候选bank、执行条件与cost；oracle拥有额外未来信息，明确不是公平可部署模型。

产物model.npz，planning.npz（source raw/split、starts、候选、learned/ideal预测/cost、两条件全部actual轨迹），
report.json（200个method-condition-attempt记录及全分母summary）与contract.json。
report含模型文件SHA256，评价结束核验未变；读取NPZ禁用pickle。
[测试](../../tests/test_s3_6_model_based_planning.py)核验候选前缀/动作范围、递归/reset隔离、
选择不调用plant、argmin/失败分期与nonfinite拒绝。

## Minimal Experiment

repo .venv Python3.12.3 / NumPy2.5.4，CPU，无新依赖/安装/数据或weights下载。
自包含训练墙模型，不需要旧checkpoint。输出目录必须不存在：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_6_model_based_planning.py --output tmp/my_s3_6
# 先预测N128→32对模型、候选前缀、最优预测cost、实际success和计算量的不同影响：
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_6_model_based_planning.py --candidates 32 --output tmp/my_s3_6_n32
```

先读contract，再看offline_validation与summary.nominal/weak_gain，最后查失败records和candidate_success_count。
`no_success_candidate_attempts`衡量当前bank缺少成功轨迹的reset数；
`misses_despite_success_candidate`衡量bank有成功轨迹但所选不成功的reset数。
单步teacher-forced误差来自400条held-out transitions；planning只接收它们的20个初态，不接收source后续state。

## Expected Result

增加嵌套候选只保证minimum **predicted cost**不升，不能保证实际cost或success单调改善。
减少N保持权重/statistics/source/starts、原候选共同前缀与zero/random实际轨迹不变；重新选择后结果可变。
unknown gain改变实际未来，learned planner仍选择同样动作，因此可能明显失败；不提前用反馈修复它。

## Actual Result

2026-10-20，CPU单BLAS线程，H10、N128，单步held-out RMSE=.373134mm：

| 方法 | nominal成功 /20 | weak_gain成功 /20 | nominal boundary / goal失败 | weak_gain boundary / goal失败 |
|---|---:|---:|---|---|
| learned | 19 | 6 | 0 / 1 | 0 / 14 |
| ideal wall-aware addition | 8 | 2 | 1 / 11 | 1 / 17 |
| zero | 0 | 0 | 1 / 19 | 1 / 19 |
| fixed random index1 | 0 | 0 | 8 / 12 | 6 / 14 |
| privileged cost oracle | 19 | 15 | 0 / 1 | 0 / 5 |

learned的nominal/weak_gain平均actual cost=.128948/.490862，bank regret=.002018/.242964。
nominal selected终点L2预测误差均值3.768mm，weak_gain为65.097mm，不能与逐坐标offline RMSE混称。
模型两条件都预测19次成功；nominal虚报0次，weak_gain虚报13次。
nominal有1个reset的bank完全没有成功候选，learned没有“有成功候选却选失败”的case；
weak_gain有5个bank无成功候选，另9个reset bank有成功候选但learned选失败。
该区分用于本bank诊断，不声称一般模型误差与候选覆盖可完全独立归因。

N32：nominal learned/oracle均13/20（7个bank无成功候选）；weak_gain learned6/20、oracle8/20，
有12个bank无成功候选，learned另2个reset选错可成功bank。
这次N增加改善名义success，但weak_gain learned仍6/20；增加搜索不等于适应执行器变化。
zero/random实际轨迹不变，但它们的regret会因oracle集合扩大而改变，不应当成基线行为改变。

最终N128运行的learned rollout+score+select p50/p95=.683442/1.592837ms，
candidate generation=1.507280/3.861850ms；N32对应.223997/.678625ms和.389721/1.073543ms。
各20次不同reset实测，warmup0；两阶段分开计时，不能把两者p95直接相加当端到端p95。
不含训练、plant/evaluator、文件I/O或真实robot观测/执行；不是部署deadline保证。
23项Stage3协议测试通过（此前18项+本轮5项）。

## Explanation / Result

learned model能在当前toy/candidate bank上支持一次动作序列选择，但模型与执行器不匹配时会失效。
它与oracle名义success相同不意味着cost排序完美或全局最优，regret仍非零。
完整独立plant执行比单步loss更接近任务评价，但依然只是合成open-loop验证。
没有真实机器人、闭环feedback、视觉语言policy、风险校准或S3.7 replanning实验。

## Failure Analysis / Failure Cases

| 问题 | 处理 / 当前边界 |
|---|---|
| 无墙模型混用于墙环境 | 基于S3.5对应train800重新拟合，版本显式记录 |
| 模型访问真实候选未来 | select只接受predictor，先选择后评价；oracle另列特权方法 |
| rollout跨reset或中途看真值 | 每bank独立s0，只有递归预测；无状态反馈 |
| 候选bank无成功轨迹 | 汇报coverage，无限扩大搜索不是本轮默认 |
| 模型排序/执行误差 | actual regret、虚报success、终点误差与真实失败分期全部保留 |
| cost最优当success最优 | 声明多项代价与二值success不同；oracle不是全局/最大成功率界 |
| 更多候选保证真实表现更好 | 只保证嵌套集合预测最小cost不升，真实环境无此保证 |
| 未知actuation gain | weak_gain learned6/20；不偷偷重拟合或提前重规划 |
| 低latency等于可部署 | 区分candidate与inference阶段，含初次调用，无真实系统deadline证据 |
| toy成功等于机器人成功 | 只测数值point-wall plant，不含接触力、惯性或真实传感器 |

## Relation to Robotics

这是model-based planning最小链路：有动作语义的候选、预测未来、任务目标和环境验证。
真正机器人还需匹配控制器、状态估计、约束与latency，并评价执行扰动；
observe→replan如何改善结果是S3.7的下一项问题，本轮不提前实现。

## Interview Capsule / Interview Notes

30秒：我训练226参数墙动力学MLP，按一次128个候选、各10步预测并用显式cost排序，执行完整所选序列。
20个held-out初态名义成功19/20，但未知执行增益.65时仅6/20，尽管离线RMSE只有.373mm。
我同时报告bank覆盖、oracle regret、虚报成功和latency；这是toy open-loop证据，不是机器人闭环成功。

2分钟：补充候选嵌套种子、train-only统计、H/T+1与world位移、known cost和learned transition分工，
解释oracle特权/非全局最优、N对预测cost与实际success的不同保证，最后讨论模型误差与执行变化。

## Likely Follow-up Questions

1. 候选生成、learned rollout、cost、argmin和actual plant分别做什么？为什么oracle不能当可部署planner？
2. 状态/动作/候选/rollout shape和单位是什么？为什么要匹配S3.5墙动力学、不能直接用S3.1模型？
3. cost包括哪些项、为什么无量纲？cost最优、success、candidate coverage与bank regret有何区别？
4. N128→32哪些数据/权重/轨迹前缀不变？对minimum predicted cost有什么保证，为什么实际success和baseline regret不必按同样方式变化？
5. 如何解释nominal19/20、weak_gain6/20和.373mm离线RMSE？怎样排查未来泄漏、候选不足与选错？为何本轮不是MPC或机器人闭环成功证明？

## Must Remember

预测未来≠真实执行；同候选公平比较；oracle是特权参照；更多搜索不修复错模型；open-loop不等于replanning。

## My Verification / Learner Verification

- [ ] Run：运行默认实验，检查五种方法、两plant条件和完整失败分母。
- [ ] Modify：先预测N128→32的不变量与不同保证，再运行对照。
- [ ] Explain：用自己的话回答五问，画清选择分支与plant/oracle评价分支。

Engineering Complete；Learning三项待用户明确报告。Current Task=S3.6；Next Task=S3.7仅计划。
STOP，不自动开始下一任务。
