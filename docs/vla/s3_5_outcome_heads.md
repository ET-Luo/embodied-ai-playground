# S3.5 — Reward / Success / Contact Prediction

## Problem / Why

World model不仅可以预测下一状态，也可以预测执行动作后的事件或reward。
本包用同一个小网络的四个输出，预测一步reward、下一时刻goal occupancy、
本次转移触墙，以及下一时刻边界越界。每个标签单独定义，不用reward符号、contact或终止标志代替success。
预计1.5小时：概念/数学25分钟、源码20分钟、Run/Modify20分钟、Explain/失败分析25分钟。
仅预测头学习，不接S3.4 latent、不实现S3.6 planning或S3.7 MPC。

## Intuition

向目标靠近可以得到正reward，却还没成功；已经在目标区，继续顶墙可以是负reward。
触墙可以发生在目标区，也可以发生在目标外；触墙本身不等于越界或任务失败。
因此需要分别建模这些结果。模型能输出0到1之间的数，也不意味着这个数是可靠部署风险概率。

## Input / Output Contract

新profile `s3_5_wall_outcomes/1.0`。它在S3.1 nominal attempted transition上增加右墙截断，
**改变了环境转移**，不能把之前的XY episodes直接当作已验证的contact数据。
所有数据重新生成，保留raw，未写回旧任务的文件或checkpoint。

| 项 | shape / dtype | 单位、frame、时序 |
|---|---|---|
| raw position | (60,21,2) float32 | world XY [m]，20转移/21状态 |
| raw action | (60,20,2) float32 | commanded world XY displacement [m]，执行t→t+1，dt=.1s |
| raw reward | (60,20) float32 | 执行后的r[t+1]，人工无量纲shaping reward |
| event_labels | (60,20,3) bool | 顺序success_next/contact_transition/failure_next |
| result | (60,) bool | episode_success，独立episode结果，非模型输入或head监督 |
| clock | (21,) float64 | 相对秒，0…2s，无terminal action |
| input | (N,4) float32→float64 | 当前XY+候选命令，仅用train统计归一化 |
| network output | (N,4) float64 | 1个标准化reward + 3个binary logits |
| final prediction | (N,) + (N,3) float64 | 反标准化reward和sigmoid分数，无量纲 |

没有language、image、force/tactile、velocity、未来state或事件输入。
固定goal/wall/边界是当前任务环境常量，不是对任意目标/墙几何的泛化模型。
数据seed350；60个独立reset，split seed351，40train/20validation，每条20转移，train800/validation400。
input/reward mean/std只fit train800；不对bool标签做归一化，不用validation调训练步数或分类阈值。

### 执行与标签的精确定义

S3.1位置相关gain与串扰先给出attempted position q，然后：

\[
q=f_{nominal}(s_t,a_t),\quad
c_{t\to t+1}=\mathbf1[q_x>0.12\,m],\quad
s_{t+1}=[\min(q_x,0.12\,m),q_y].
\]

本contact标签是右墙主动截断命令的几何事件，包括持续顶墙，不只首次碰撞。
它不是force测量、碰撞冲量、MuJoCo接触求解或现实传感器标签；等于墙位置而未超出不算active contact。
只截断x，y不因接触改变，没有弹性/摩擦/惯性动力学。

定义固定目标g=[.09,0]m、半径.045m，distance为world XY欧氏距离：

\[
u_{t+1}=\mathbf1[\|s_{t+1}-g\|_2\le .045\,m],\qquad
v_{t+1}=\mathbf1[s_{t+1,x}<-.20\,m\ \lor\ |s_{t+1,y}|>.20\,m],
\]
\[
r_{t+1}=\frac{\|s_t-g\|_2-\|s_{t+1}-g\|_2}{.04\,m}
-.2c_{t\to t+1}-.5v_{t+1}.
\]

- `success_next`=u：**下一时刻是否在目标区**，不是“从现在最终一定成功”。
- `contact_transition`=c：本次命令是否被墙主动截断，使用attempted q定义。
- `failure_next`=v：下一时刻是否违反指定左侧/y边界，是局部failure event，不是整段未来风险。
- reward独立采用progress与contact/failure惩罚；不从success bool直接复制。
- `episode_success = u_T AND no v_1…v_T`：只作collection结果保存，绝不复制给每个transition或输入网络。

“独立标签”指各有明确语义与计算来源，**不表示统计独立**。它们共享执行结果，可能相关或重叠。
label均由执行后的真值计算，作为监督合法；预测时只能使用执行前s_t与候选a_t。
即使当前state已越界，也保留所有转移；v是下一状态事件，不要求首次进入。

### 数据采集与边界

每条episode初态均匀采样x∈[-.18,.115]m、y∈[-.22,.22]m；初态可能已在y越界区域，明确保留。
episode index%4选择random exploration、固定goal反馈、持续右推、沿初始y符号向外四种手写行为，
增加正负标签覆盖。它们只用于数据采集，**不是本轮训练的policy**。
动作各模式范围见脚本，最大绝对轴命令约.06m；goal模式含±.005m扰动。
四种行为都整episode划分，不因success/failure/contact删样本。
所有episode只在20步时限截断；事件不触发termination，因此越界之后仍有数据。

OOD `action_scale3`：同400个validation当前状态，命令×3，重新执行墙toy并重算四标签。
它是配对单步反事实，不是连贯OOD rollout，也不修改原始validation数据。

## Core Concepts

- Reward head回归实数；其他三个head是各自的Bernoulli监督，不能用softmax强迫它们互斥。
- 共享32维tanh trunk，输出层四列实现独立读出；共享feature不意味着标签等价。
- 本failure head预测一步边界事件；最终任务失败还依赖未来动作、反馈、控制器和终止规则。
- Sigmoid是有界分数；训练BCE不会自动提供分布变化下的概率校准保证。
- FP/FN对不同head含义不同：success FP虚报到达，contact FN漏掉触墙，failure FN漏掉越界。
- 只报accuracy容易掩盖类别频率及错误代价，因此保留完整分母、正样本数、precision/recall及Brier/BCE。

## Mathematics

输入x=[s_t,a_t]，标准化仅用train800的逐通道mean/std；h=tanh(x_norm W1+b1)，
4→32→4共292参数，output第一列为reward，后三列为success/contact/failure logits。

\[
\hat r=\mu_r+\sigma_r o_0,\quad p_j=\sigma(o_j)=1/(1+e^{-o_j}),\ j=1,2,3.
\]
\[
L=\frac1N\sum_n(o_{n,0}-\tilde r_n)^2+
\frac1{3N}\sum_{n,j}[\log(1+e^{o_{n,j}})-y_{n,j}o_{n,j}].
\]

这是标准化reward MSE + 三head平均BCE，未做class weighting或验证集loss调权。
稳定实现用`logaddexp(0,logit)`和按绝对值分支的sigmoid，避免exp溢出。
reward输出梯度2*(prediction-target)/N，logit梯度(p-y)/(3N)；再经线性层/tanh手写反向。
全batch Adam lr=.01、β=.9/.999、ε=1e-8，默认2500步，seed352。

训练loss无量纲；报告reward MAE/RMSE同样无量纲（shaping分数），不能标成m或m²。
分类decision=`p>=threshold`，默认threshold=.5；这是预先声明的诊断阈值，没有用validation/OOD选择。
Brier=mean((p−y)²)，BCE=mean负log likelihood，均不依赖分类阈值。
precision=TP/(TP+FP)，recall=TP/(TP+FN)；零分母返回null，不伪称0或1。

threshold .5→.2且分数不变时，预测positive集合只能扩大：FP不减少、FN不增加；
reward、概率、Brier/BCE保持不变。更低FN不等于整体更好或更安全，代价需由具体任务决定。
本包不做probability calibration、risk bound或动作拒绝部署。

## Architecture / Data Flow

`current state + collection command → nominal attempted state → right-wall clip → next state`

`attempted/next state → separately defined contact/success/failure/reward supervision`

`whole-episode split → train-only normalization → 4→32 tanh shared trunk → four output columns`

`validation current state + same recorded command → reward / three sigmoid scores → per-head evaluation`

next_state、contact、event labels、episode_success只属于监督/结果分支，绝不拼入输入。
本轮直接学习outcome heads，不先预测next_state再用规则转标签；不隐式复用S3.4或假装joint world model。

## Math / Architecture → Code / Implementation

[脚本](../../scripts/s3_5_outcome_heads.py)：`step`实现墙约束，`annotate`按真实结果定义标签，
`dataset`保留连续episode，`inputs`只拼当前state/command；`loss_and_gradients`显式多任务loss/梯度，
`fit`只接收train，`predict`不接收label/next_state，`binary_metrics`计算每个head的错误与分数误差。

基线：reward用train平均值，分类用train正样本比例作为常量分数；它们不借validation prevalence。
动作消融把输入的物理命令置0再正确标准化，保持真实标签不变，测量动作信息依赖，不能当新zero-action真实结果。

产物model.npz（权重/统计）、episodes.npz（raw、全部标签、episode结果、split/行为/clock/boundary），
evaluation.npz（validation/OOD输入、真值reward/标签、预测reward/分数），report.json和contract.json。
NPZ读取使用`allow_pickle=False`。
[测试](../../tests/test_s3_5_outcome_heads.py)检查标签组合、执行后时序、墙截断、阈值单调性、
每个输出头梯度与极端logits稳定性；运行前另有每参数tensor活跃坐标有限差分检查。

## Minimal Experiment

repo .venv Python3.12.3 / NumPy2.5.4，CPU，无新依赖/安装/数据或weights下载。
不依赖旧checkpoint；以下目录必须不存在：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_5_outcome_heads.py --output tmp/my_s3_5
# 先预测threshold .5→.2哪些输出不变、哪些错误只能向一个方向变化：
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_5_outcome_heads.py --threshold .2 --output tmp/my_s3_5_t02
```

两条命令确定性重训同一模型，threshold不参加fit，权重/数据/分数应逐值一致。
先查contract的标签时间，再比较report validation的learned/train_constant/zero_action_ablation；
尤其分清per-transition success_next与独立episode_success，不能把分类accuracy当robot success rate。

## Expected Result

名义held-out结果预计优于train常量reward/分数基线，但保留未学会的head错误。
OOD动作放大可能显著降低效果，不能由名义BCE推断其鲁棒性。
降低threshold通常换取更少FN与更多FP；不保证每个head的FN严格减少或FP严格增加。

## Actual Result

2026-10-20，CPU单BLAS线程，train800/validation400：
reward train/validation RMSE=.078729/.152957；validation train-mean基线=.735068、zero-action消融=.481000。
action×3反事实reward RMSE=1.090921，常量基线1.837897、zero-action消融1.537491。
这些是无量纲reward分数误差，不是位置误差。

| 条件 / head | positive / N | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|
| validation success_next | 72/400 | 46 | 10 | 26 | 318 |
| validation contact_transition | 131/400 | 129 | 0 | 2 | 269 |
| validation failure_next | 138/400 | 132 | 1 | 6 | 261 |
| action×3 success_next | 73/400 | 44 | 31 | 29 | 296 |
| action×3 contact_transition | 155/400 | 150 | 1 | 5 | 244 |
| action×3 failure_next | 156/400 | 112 | 2 | 44 | 242 |

validation success/contact/failure Brier=.063394/.003827/.018685，BCE=.208748/.011542/.080583；
action×3 Brier=.128641/.012416/.096849，BCE=.718467/.045675/.472418。
train正样本比例=.27875/.12625/.2825；有限episode划分导致validation比例不同，尤其contact，
不能把400个相关转移当成400次独立任务试验。

threshold .2时validation三head TP/FP/FN/TN变为60/10/12/318、130/0/1/269、132/5/6/257；
OOD failure FN44→33，FP2→3。输入、标签、权重、预测reward、概率、Brier/BCE不变。
没有根据这些结果推荐部署阈值；这是预设学习对照。

默认batch1全head推理p50/p95=.008156/.008616ms，20warmup/200samples，
含input标准化、MLP、reward反标准化和sigmoid；不含I/O、sensor、执行、规划或标签计算。
四tensor梯度检查最大绝对误差约1.126e-11，额外测试核验四个head各自bias梯度。
18项Stage3测试通过（此前13项+本轮5项）。

## Explanation / Result

训练可拟合这类synthetic一步标签，但success head仍有明显漏报，OOD failure漏报更严重。
contact是几何约束事件，不是learned真实接触物理；failure分数不提供校准风险保证。
本任务没有训练或评价head驱动的policy、planner或真实机器人控制成功；
collection手写行为的episode结果只是数据字段，不是预测head带来的闭环success证据。

## Failure Analysis / Failure Cases

| 问题 | 处理 / 限制 |
|---|---|
| reward>0就当success | 正progress尚未到目标；success同时contact可得到负reward，测试明确覆盖 |
| contact就当failure | contact可在goal内或外；failure按独立边界定义 |
| 把episode结果复制到每步 | episode_success单独保存，只监督执行后一步事件 |
| 用未来state/force混入输入 | `inputs/predict`只接受当前state与command；无force或未来观测字段 |
| 沿用旧point数据声称contact | 新wall profile显式改动力学，重新生成，不混用schema |
| 事件出现就删轨迹/终止 | 明确time-limit-only协议，事件后也保留，记录全部FP/FN |
| sigmoid当可靠risk | 仅BCE/Brier评价，无概率/安全保证，尤其OOD可漏报 |
| 类不平衡只看accuracy | 同报positive count、TP/FP/FN/TN、precision/recall及分数误差 |
| threshold偷看validation选优 | 默认.5/对照.2预先声明，threshold不参与fit，不推荐部署最优值 |
| zero-action消融和真实执行混淆 | 保留原标签只改模型输入，属于信息消融，不是新动作的物理评价 |
| 指定OOD性能代表泛化 | 仅action×3反事实，没有新wall/goal/真实contact或未建模机制评价 |

## Relation to Robotics

机器人规划可能使用reward/success/contact预测，但应先声明预测时间范围和failure定义。
一步边界事件与未来任务失败、接触力超限是不同问题；后续规划和闭环需要单独实测，
本包不据这些head得分执行动作或自动进入S3.6。

## Interview Capsule / Interview Notes

30秒：我用292参数NumPy多头网络，从当前world XY和位移命令预测一步reward及success/contact/failure事件。
标签来自新版本墙约束toy，success是下一状态goal occupancy，contact是命令被墙截断，failure是下一状态越界。
完整episode留出，train-only标准化，各head报告FP/FN；动作×3时failure漏报44/156，不能将sigmoid当可信风险。

2分钟：补充T/T+1时序、attempted与executed状态、四种标签的独立定义、回归+3BCE的梯度与单位，
区分一步标签/episode结果、阈值移动/Brier不变、常量和动作消融，最后说明缺真实接触/闭环证据。

## Likely Follow-up Questions

1. 四个输出各自预测什么时间范围？success_next、episode_success、failure_next、terminated/truncated为何不能互换？
2. 写出墙step和reward公式；举例说明正reward不等于success、contact不等于failure，标签独立定义为何不等于统计独立？
3. 模型输入/output shape、单位和loss是什么？为何三个binary heads用sigmoid/BCE而不是softmax？next_state作为label来源为何不是输入泄漏？
4. threshold .5→.2时哪些产物不变？FP/FN有何单调性，Brier/BCE为何不变？为何不能称failure分数为校准风险？
5. 如何解释train/validation/OOD的结果和动作消融？为什么低离线loss、contact高accuracy或采集episode成功不证明head驱动的机器人闭环成功？

## Must Remember

reward≠success≠contact≠failure；一步标签≠episode结果；sigmoid≠校准风险；输入仅当前状态/候选命令；全分母误报漏报。

## My Verification / Learner Verification

- [ ] Run：运行默认实验，查标签时序、reward误差和三个head完整混淆矩阵。
- [ ] Modify：先预测threshold .5→.2的不变量及FP/FN方向，再运行核对。
- [ ] Explain：用自己的话回答五问，画执行/标签分支与因果预测分支。

Engineering Complete；Learning三项待用户明确报告。Current Task=S3.5；Next Task=S3.6仅计划。
STOP，不自动开始下一任务。
