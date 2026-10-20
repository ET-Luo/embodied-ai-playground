# S3.2 — Multi-step Rollout

## Problem / Why

S3.1 单步验证RMSE约0.138mm，但每次输入都是真实当前状态。
若模型必须把自己的预测继续当输入，20步或100步后还准吗？本任务冻结S3.1模型，
比较teacher-forced（TF，真实状态条件）与autoregressive（AR，递归预测）的误差。
只做rollout评价，不训练新模型、不实现ensemble、planning或MPC。
预计1小时：Problem→Mathematics 15分钟、源码15分钟、Run/Modify 15分钟、Explain 15分钟。

## Intuition

TF像每走一步都拿到准确地图位置，AR像只知道起点、以后一直用自己的估计推算。
TF能够评价每个局部预测，却掩盖上一步预测误差传入下一步的影响。
未来动作在两种模式中相同且事先固定；本例没有根据预测位置重新计算动作的policy。

## Input / Output Contract

继承 [S3.1](s3_1_state_dynamics.md) 的 `s3_1_xy_command_dynamics/1.0` 物理语义，
新输出profile `s3_2_rollout/1.0`，详见每次生成的contract.json。

| 项目 | shape / dtype | 物理/时间含义 |
|---|---|---|
| checkpoint | W1(4,32), b1(32), W2(32,2), b2(2) float64 | 冻结226参数及原train-only统计 |
| raw truth | (10,H+1,2) float32 | 每条原validation episode的world XY位置[m] |
| raw action | (10,H,2) float32 | world XY commanded displacement[m]，不是实际位移 |
| clock | (H+1,) float64 | 相对秒，dt=.1s，默认H100即10秒 |
| TF prediction | (10,H,2) float64 | 第t槽预测s[t+1]，输入真实s[t] |
| AR prediction | (10,H+1,2) float64 | 第0槽真实起点，其后只有递归预测 |
| MLP input | (10,4) float32 | `[x_t,y_t,a_x_t,a_y_t]`，每步重新按冻结统计归一化 |
| MLP output | (10,2) float64 | 标准化实际位移→物理位移→加当前输入位置 |

无视觉、语言、速度、力或接触。使用S3.1原来的10条held-out episode，未重新划分。
前20个动作/21个真实状态逐值保留；随后每episode使用独立seed `3200+原episode_id`
采样[-.04,.04]m动作，由S3.1确定性transition产生延长部分真值。
因此每个episode从20步延长为H步，只有t0 reset，不拼接其他episode，不从第21步重新reset。
这是对已知synthetic generator的可重复延长，不是为真实日志的截断点补造观测。
新动作与预测器无关，不受bias参数影响；H改变时所有共有前缀保持相同。
原source文件只读，SHA256记录在report。不存在source时明确失败，不隐式重训或下载。

数值约定：MLP拼接输入显式转float32，沿用S3.1标准化精度；网络与最终状态计算float64。
每次AR继续推理时重新把输入转float32。TF/AR基线统一在float64算加法；oracle遵循generator的float32转移。
不能把这些精度差异误当成物理随机性。所有raw物理数组原样保存，不写回source。

## Core Concepts

- 本包的TF是**评价时**每步使用真实状态，不是重新训练或使用未来状态预测动作。
- AR是状态预测递归；不是Stage2逐token生成action的AR解码，二者递归对象不同。
- 在第h步，TF只预测真实s[h−1]到s[h]；AR从s0预测到s[h]，未获得中途真值。
- h=1输入完全相同，TF与AR应逐值相等；h>1误差不保证单调上升，也不必AR永远比TF大。
- 全程固定动作；这里没有环境反馈给policy、没有closed-loop任务success定义。

## Mathematics

冻结预测器 \(\hat f_\theta(s,a)\) 是S3.1 residual MLP恢复物理单位后的下一状态函数。

\[
\hat s^{TF}_{t+1}=\hat f_\theta(s_t,a_t),\qquad
\hat s^{AR}_0=s_0,\quad
\hat s^{AR}_{t+1}=\hat f_\theta(\hat s^{AR}_t,a_t).
\]

所有s、a均为2维world XY向量，单位m；t对应.1s的离散转移。
设局部模型误差 \(\epsilon_t=\hat f_\theta(s_t,a_t)-f(s_t,a_t)\)，
递归状态误差 \(e_t=\hat s_t^{AR}-s_t\)。有恒等分解：

\[
e_{t+1}=\hat f_\theta(\hat s_t^{AR},a_t)-\hat f_\theta(s_t,a_t)+\epsilon_t.
\]

局部线性化可写 \(e_{t+1}\approx J_s\hat f_\theta\,e_t+\epsilon_t\)。
这解释误差传递、放大或抵消；不是证明本模型稳定的定理，本实验没有估计Jacobian谱。
不能普遍声称误差严格线性、指数增长或每步更差。

**度量分母**：h处端点RMSE为10个episode×2坐标误差平方的均值再开方；
TF local也用相同h处10个episode，但输入含第h−1步真实状态。
prefix RMSE则覆盖10×h×2坐标，不含t0，不能与端点RMSE混称。
所有MSE单位m²、RMSE单位m，表中转换为mm；p95 L2是每个预测点的2D误差范数95分位。

**偏置探针**：在MLP反标准化后，每次额外加 \(b=[\beta/1000,0]\)m，默认β=1mm。
这是人工改坏的预测器，不是重新训练或执行器扰动。非线性MLP误差不必与β成正比。
另外在零动作/常状态的解析系统中，令 \(\hat f(s,0)=s+b\)，则
TF L2误差=|β|mm，AR第h步L2误差=h|β|mm；此处严格线性仅属于这个解析探针。
若只x轴有误差，每坐标RMSE=L2/√2，不能把1mm的x偏置当作1mm的二维元素RMSE。

## Architecture / Data Flow

`S3.1 contract + checkpoint + raw episodes → validate semantics/split/train statistics → freeze hashes`

`held-out first20 transitions → independent fixed action extension → true trajectory`

`TF: truth[t] + action[t] → frozen predictor → predicted next state`

`AR: previous prediction + same action[t] → frozen predictor → next prediction`

`→ per-horizon endpoint/prefix metrics + full curves + training-box audit`

predictor拿不到后续真实状态，oracle是单独的特权参考；模型预测后不被oracle校正。

## Math / Architecture → Code / Implementation

[脚本](../../scripts/s3_2_multistep_rollout.py)：
`load_source`检查契约、shape/dtype/finite、clock、完整split partition、真实transition对齐和train-only统计；
`predict`只复用S3.1 `forward`，不训练；`extended_episodes`构造固定延长动作；
`evaluate`用独立TF/AR数组保持两条数据流；`summarize`区分端点与前缀分母。
不改S3.1源代码，不导入旧仓库。

输出 `rollouts.npz`（所有真值、动作及五种预测器的TF/AR完整轨迹），`curves.csv`
（每步TF local和AR endpoint RMSE，单位m），`report.json`和`contract.json`。
NPZ用`np.load(..., allow_pickle=False)`读取。

协议测试 [test_s3_2_multistep_rollout.py](../../tests/test_s3_2_multistep_rollout.py)
独立于训练checkpoint：修改未来truth不会改变AR、精确oracle无漂移、已知偏置累积和episode reset、
延长时保留原source及全部共有前缀。它们验证评价语义，不是机器人成功测试。

## Minimal Experiment

已有repo .venv Python3.12.3 / NumPy2.5.4，CPU，无新依赖、安装、数据或权重下载。
默认读取 `tmp/s3_1_final`；若你用自己的S3.1输出，用`--checkpoint tmp/my_s3_1`。
缺少checkpoint时，请先按S3.1命令生成，S3.2不会自动重训。
以下输出目录必须不存在：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_2_multistep_rollout.py --output tmp/my_s3_2
# 先预测bias从1→2mm对数据/hash/未偏置模型/偏置模型的不同影响，再运行：
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_2_multistep_rollout.py --bias-mm 2 --output tmp/my_s3_2_bias2
# 可选：先预测缩短时域是否改变前20步，再做40步对照
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_2_multistep_rollout.py --steps 40 --output tmp/my_s3_2_h40
```

先看report的`s3_1_replay`应恢复原200条单步误差，再看`evaluation.mlp.20/100`。
五种预测器是persistence、ideal_addition、MLP、MLP+人工偏置、精确oracle，全部保留10条episode。
steps限制21…500，bias限制有限[-10,10]mm。CLI拒绝非法参数和已有输出目录，不覆盖产物。

## Expected Result

h1时TF=AR；oracle全部为0。递归预测可能积累误差，100步误差预计大于局部单步误差，
但不是数学保证。bias变化只影响人工偏置预测器和解析探针，其他误差与轨迹应保持不变。
零动作解析探针β1→2mm严格使100步L2漂移100→200mm；真实MLP不要求严格翻倍。

## Actual Result

2026-10-20，CPU单BLAS线程，冻结S3.1 2000步checkpoint：

| h / 物理时间 | MLP TF local RMSE [mm] | MLP AR endpoint RMSE [mm] | MLP+1mm AR endpoint [mm] |
|---|---:|---:|---:|
| 1 / .1s | .0983 | .0983 | .6407 |
| 5 / .5s | .0924 | .3999 | 3.2858 |
| 20 / 2s | .1042 | 1.6074 | 12.7704 |
| 100 / 10s | .4649 | 15.5763 | 60.6893 |

这里h1只含每episode第一个转移，故不等于S3.1全部200条转移的0.137870mm。
前20步TF prefix RMSE精确复现S3.1的0.137870mm；AR prefix RMSE为0.943050mm，
与20步端点1.607353mm也是不同分母。
100步不动/理想加法AR端点RMSE为175.8165/71.8391mm，oracle TF/AR均为0。
MLP100步prefix RMSE为7.598845mm，端点p95 L2为40.638241mm（仅10个episode，小分母描述统计）。

bias2mm保持所有未偏置结果不变，MLP+偏置100步端点RMSE为128.2071mm，非精确翻倍。
解析零动作探针TF L2 1→2mm，100步AR L2 100→200mm；它不是learned结果。
H40对照MLP端点RMSE3.9189mm，前20步及所有共有轨迹前缀保持不变。

100步输入中，truth与MLP AR各16.3%状态在原train输入状态axis-aligned包围盒外；
20步对应2.0%和2.5%。盒内不代表统计ID，盒外也不等于已校准风险；此诊断不是S3.3 uncertainty。
长时域的误差混合了递归传递与状态分布变化，不能仅归因于步数。

首轮batch10执行全部100步TF+AR共200次批量预测p50/p95约2.596/4.159ms，3warmup/20samples；
含Python循环、finite及h1一致性核验，不含生成真值、metric、文件I/O、环境或planning，非机器人端到端latency。
四项协议单元测试通过；另核验artifact重载、源hash未变、bias不变量、H前缀不变量和CLI拒绝行为。

## Explanation / Result

小单步误差仍可伴随更大的多步漂移。本模型同一初态、同一动作下可做短时近似，
但100步误差比真实状态条件的局部预测大得多。没有训练新模型、调验证集超参或声称修复漂移。
没有policy、任务成功标签或真实机器人闭环，success/failure stage未测。

## Failure Analysis / Failure Cases

| 失败/误读 | 本任务处理与限制 |
|---|---|
| AR中途取真实state | 未来truth反事实测试：AR轨迹不变；TF发生变化 |
| 不同模型用不同动作 | 全预测器共用同一raw动作数组，bias不改变动作 |
| flatten后跨episode递推 | 每条episode独立起点，10条并行，没有跨reset连接 |
| 单步误差乘H当普遍定理 | 只在解析恒偏置系统严格成立，MLP可能放大/抵消 |
| 把TF h100称100步预测 | TF h100只是以真实s99为条件的一步预测 |
| 把端点与prefix RMSE混用 | 两组指标同时命名、显式分母，无t0稀释 |
| 100步依然全为ID | 报训练状态盒外比例，不能分离累积误差与外推影响 |
| 平均掩盖坏episode | 保存全部轨迹、p95 L2；十episode不足以证明总体鲁棒性 |
| 真实日志截断后补真值 | 只延长本toy已知generator，不能照搬到真实机器人数据 |
| 低latency=部署成功 | 只计CPU推理循环，未测环境、规划或任务闭环 |

## Relation to Robotics

若规划器对预测未来打分，长时预测失真会导致错误动作选择。但本包仅评价预测器，
不提前实现MPC；重新观测是否改善任务成功需要后续匹配闭环实验验证。

## Interview Capsule / Interview Notes

30秒：我冻结一个226参数动力学MLP，对相同held-out初态和固定动作比较TF与AR。
TF每步输入真实状态，AR只在起点看真值。原单步RMSE0.138mm，但20/100步AR端点约1.61/15.58mm。
长时状态部分超出训练范围，不能把漂移全归因于递归。这里只是toy预测，未证明机器人控制成功。

2分钟：补充误差递推分解、端点和prefix分母、h1一致性、oracle零误差、未来truth反事实、
恒偏置解析probe和源checkpoint hash；说明train-only normalization、无重训和完整episode边界。
最后讨论分布变化、误差抵消、同动作比较，以及真实系统还需接触/隐藏状态/闭环评价。

## Likely Follow-up Questions

1. 写出TF与AR递推公式；为什么h1相同，而TF在h100仍只是在做局部单步预测？
2. S3.1 RMSE0.138mm、h1的0.0983mm、h20 AR端点1.607mm分别用了哪些样本/分母？
3. 用误差递推解释局部误差如何影响以后状态；为什么不能保证误差严格线性或单调增长？
4. bias1→2mm时哪些artifact与结果应不变？零动作解析系统的100步L2漂移是多少？MLP为何不必翻倍？
5. 如何检查未来真值泄漏、动作不匹配和跨episode递推？训练状态盒外16.3%说明什么、不说明什么？

## Must Remember

TF输入真值；AR输入预测；同初态同动作；端点≠prefix；小单步误差≠长时稳定≠机器人成功。

## My Verification / Learner Verification

- [ ] Run：运行默认实验，读TF/AR轨迹、20/100步端点与S3.1 replay指标。
- [ ] Modify：先预测bias1→2mm，再比较不变量与偏置预测器；可选H40前缀实验。
- [ ] Explain：用自己的话回答五问，画出两条数据流并说明长时误差的证据边界。

Engineering Complete；Learning待用户明确报告。Current Task=S3.2；Next Task=S3.3仅计划。
STOP，不自动开始ensemble或下一任务。
