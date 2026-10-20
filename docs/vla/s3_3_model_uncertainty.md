# S3.3 — Model Uncertainty

## Problem / Why

S3.2 说明单步误差小不保证多步预测稳定。下一问：当模型不知道自己是否预测准确时，
多个独立训练成员的分歧是否能提醒我们？本包训练小ensemble，比较名义分布和指定OOD探针，
同时构造“分歧很小、错误很大”的反例。只评价单步预测，不实现风险head、planning或MPC。
预计1.5小时：概念/公式25分钟、源码20分钟、Run/Modify20分钟、Explain25分钟。

## Intuition

请五位学习过相同规律、但见过不同训练episode组合的人预测结果。
他们意见不同时，值得进一步检查；他们意见相同时，也可能共享同一种盲点。
复制同一个模型五次不会增加知识；没有输入到模型的执行机制变化，也可能对所有成员不可见。

## Input / Output Contract

名义物理规律沿用 [S3.1](s3_1_state_dynamics.md)：位置相关执行gain和XY串扰，
当前位置与commanded world displacement决定下一位置。新profile `s3_3_ensemble_xy/1.0`，
每次生成contract.json明确声明评估条件；不写回或修改旧raw数据。

| 项目 | shape / dtype | 语义、frame与时间 |
|---|---|---|
| 原始名义episodes | state (40,21,2) / action (40,20,2) float32 | world XY位置[m]、命令位移[m]，dt=.1s |
| train | 30 episodes / 600 transitions | seed32划分，完全沿用S3.1 generator/split |
| calibration | 10新episodes / 200 transitions | seed330，独立reset与动作；只设置score阈值 |
| ID evaluation | 10 held-out episodes / 200 transitions | 从未参与梯度、normalizer或阈值拟合 |
| bootstrap IDs | (K,30) int64 | 每成员从30个train IDs有放回抽取30次 |
| model input | (N,4) float32 | `[x_t,y_t,a_x_t,a_y_t]`，真实当前状态+命令 |
| model target | (N,2) float32 | 实际位移s[t+1]−s[t]，train统计标准化 |
| weights | (K,4,32)/(K,32)/(K,32,2)/(K,2) float64 | K个独立4→32→2 MLP，默认K5，共1130参数 |
| member predictions | (K,N,2) float64 | world XY下一位置[m]，已反标准化 |
| ensemble mean | (N,2) float64 | 成员下一位置的算术平均[m] |
| disagreement / error | (N,) float64 | score单位m；error为预测均值与真值的2D L2[m] |

名义episodes有20次转移/21个状态，最后按时限截断，无success标签。仅位置/动作模态，
没有language、image、速度、force、contact或过程噪声。clock为float64相对秒。
原train600条转移拟合一套共享x/delta mean/std；bootstrap只用于各成员梯度训练。
calibration/evaluation/OOD不参与normalizer、成员训练或checkpoint选择。
S3.3自包含重建确定性数据并训练新ensemble，不依赖ignored S3.1 checkpoint，也不替换它。

### 四个评价条件

每组200个一阶转移tuple，按原10条held-out episode逐步配对。三个探针均为**counterfactual单步tuple**，
不是连贯的新OOD rollout；原episode/timestep索引用于配对，不代表新轨迹连续。

| 条件 | 当前状态/动作 | 下一状态真值来源 |
|---|---|---|
| ID | 原held-out状态、动作 | 原真实s[t+1]，名义生成分布留出集 |
| state_shift | 当前位置x+.8m，动作不变 | 同名义transition作用于新输入 |
| action_scale | 状态不变，命令两轴×3 | 同名义transition作用于新输入；动作范围扩至±.12m |
| hidden_mechanism_shift | **与ID完全相同的输入** | 原下一位置x额外+βmm，默认β10 |

hidden_mechanism_shift显式改变了ground-truth转移机制，不改动作的命令语义，
也不是给预测值加偏置。可以理解为未提供给模型的执行机制变化，但不是真实机器人实测。
此时单一名义机制假设不再覆盖部署条件；模型没有机制标识，不能从相同输入识别其变化。
ID表示同名义生成分布，不保证每点都在train有限样本包围盒内；OOD仅指本包明确的三个分布/机制对照。

## Core Concepts

- Ensemble成员多样性来自不同随机初始化与**episode bootstrap**；共享架构/数据仍会产生相关错误。
- Epistemic（知识不足）可由分歧尝试诊断，但本分数不是严格的Bayesian posterior uncertainty。
- Aleatoric（给定输入后的不可约随机性）与成员差异不同。本toy无随机转移，未学习噪声方差head。
- 输入范围改变是covariate shift；相同输入的真实转移改变是mechanism/concept shift，两者可检测性不同。
- Calibration集仅用来固定分数百分位阈值；名字不意味着获得“失败概率校准”。
- 本实验的高误差是L2>5mm的人工诊断标签，不是机器人失败、碰撞概率或安全风险。

## Mathematics

每成员预测标准化实际位移，恢复下一位置：

\[
\hat s'_{k}=s+\mu_{\Delta}+\sigma_{\Delta}\odot
\operatorname{MLP}_{\theta_k}(([s,a]-\mu_x)/\sigma_x).
\]

参数/mean/std与 [S3.1](s3_1_state_dynamics.md) 一致，训练为全batch Adam、lr=.01、
β1=.9、β2=.999、ε=1e-8，默认1500步，K个成员各226参数。
bootstrap每次采样保留完整20次转移，重复episode重复计入梯度；不跨reset拼接状态转移。
种子data31/split32/calibration330；成员bootstrap3300+k、初始化3400+k。

对一个输入，成员预测形状(K,2)，均值与score：

\[
\bar s'=\frac1K\sum_k\hat s'_k,\qquad
D=\sqrt{\frac1{2K}\sum_{k=1}^{K}\sum_{d=1}^{2}
(\hat s'_{k,d}-\bar s'_d)^2}.
\]

这是成员与坐标的population RMS spread，ddof=0，单位m。
不是ensemble均值的standard error，不能再除√K当作可信误差界。
真值不进入D；真实L2误差 \(E=\|\bar s'-s'\|_2\) 只在离线评价时可用。
报告RMSE则是N×2坐标的元素RMSE，不等于L2均值或score。

阈值 \(\tau=Q_{.95}(D_{calibration})\)，200条独立名义calibration转移的分歧95分位；
`flag = D > τ`；诊断标签`bad = E > .005m`。
TP/FP/FN/TN在每组全部200条上计算。高flag率不代表高error率；
阈值仅限制这次calibration样本的相对排序，不保证ID假阳性率5%，更不保证OOD风险。
每个reset内转移相关，不能把200个点当200次独立机器人试验。

**盲点**：当所有成员预测共同偏差c时，
\(D(\hat s'_1+c,\ldots,\hat s'_K+c)=D(\hat s'_1,\ldots,\hat s'_K)\)，
但相对真值的误差可能增大。复制同一预测K次，D=0（数值舍入除外），也可能所有成员都错。
改变target而保持输入/权重不变，同样不会改变D。

## Architecture / Data Flow

`S3.1 generator → episode split → train-only shared statistics`

`train episode bootstrap × K + independent initialization → K residual MLPs → frozen ensemble`

`independent calibration inputs → member predictions → disagreement q95 → fixed threshold`

`ID / paired OOD inputs → member predictions → mean + score → compare held-out truth`

`→ per-member/ensemble physical error + flagged counts + diagnostic confusion + complete sample records`

所有truth只进入梯度训练标签或离线评价；评价target不进入score函数。
机制变化条件与ID共用完全相同的预测和score，只有truth不同。

## Math / Architecture → Code / Implementation

[脚本](../../scripts/s3_3_model_uncertainty.py)：`calibration_episodes`独立生成阈值集，
`flatten`只取episode内转移，`fit_statistics`只接收train600，`train`复用S3.1显式forward/gradients；
`predict_members`逐成员恢复物理next state，`disagreement`实现2K分母，
`summarize`同时报告每成员/均值误差、score分位数、score/L2 Pearson相关与误报漏报。
相关系数无量纲，仅每组样本描述，非因果/概率校准证据；常量score时返回null。

产物：`episodes.npz`保留raw/split/calibration，`ensemble.npz`保存权重/共享统计/bootstrap IDs，
`evaluation.npz`保存每组原始输入/真值/所有成员预测/均值/score/error，
`samples.csv`保存逐样本诊断，另有report.json与contract.json。读取NPZ禁用pickle。
state/action OOD目标重新调用名义generator，不裁剪新输入、套用ID下一状态或修改原始真值。

[协议测试](../../tests/test_s3_3_model_uncertainty.py)核验分母/单位、共同偏置与成员排序不改score、
相同预测无法识别target变化、严格阈值的混淆矩阵，以及NaN/单成员/错误rank拒绝。
不以测试通过代替模型评价或学习掌握。

## Minimal Experiment

repo .venv Python3.12.3 / NumPy2.5.4，CPU、模型计算float64；无需新依赖、安装或下载。
从仓库根目录运行，目录必须不存在。每条命令确定性重训同样的ensemble，通常几秒：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_3_model_uncertainty.py --output tmp/my_s3_3
# 先预测隐藏机制偏移10→20mm对权重、threshold、score与error的不同影响，再运行：
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_3_model_uncertainty.py --shift-mm 20 --output tmp/my_s3_3_shift20
# 可选：先预测成员减少会改变什么；不保证score/误差单调变化
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_3_model_uncertainty.py --members 2 --output tmp/my_s3_3_k2
```

先查看contract，再读report的score_threshold_m、四组evaluation与cloned_member_control。
flag不是拒绝执行动作；本包不部署任何过滤器。

## Expected Result

指定的state/action范围偏移预计增加误差与分歧，但不保证每个OOD样本高score。
机制变化条件应与ID score逐值一致；偏移10→20mm只改变其target与相关误差/diagnostic labels，
以及使用同一target的复制成员对照。checkpoint、raw、calibration、threshold及全部预测不变。
复制成员的score约0，但可能高误差。增加成员数不保证更准或更可信。

## Actual Result

2026-10-20，CPU单BLAS线程，K5、1500steps/member；threshold=.177805mm：

| 条件（各200条） | 均值预测RMSE [mm] | score中位数 [mm] | flagged | L2>5mm | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---|
| ID | .125052 | .065647 | 1 | 0 | 0/1/0/199 |
| state x+.8m | 2.946716 | 1.251255 | 200 | 49 | 49/151/0/0 |
| action ×3 | 9.238062 | .651191 | 168 | 110 | 110/58/0/32 |
| hidden next-x+10mm | 7.068643 | .065647 | 1 | 200 | 1/0/199/0 |

ID五个成员RMSE为.142001/.123439/.124462/.193424/.144547mm；
均值不保证胜过每个成员，本次第二/第三成员略好于ensemble均值。
score/L2 Pearson相关分别约.6451/.7765/.8012/.00365；相关性不等于failure probability。
calibration 200条中10条超过q95；ID evaluation只1条，不是固定5%保证。
state shift flagged200条而高误差49条，说明score flag与5mm阈值语义不同。
复制第一个成员K次并对隐藏机制真值评价，score中位数0，200/200高误差全部漏报。

shift20mm对照：其他三组完全不变；hidden RMSE14.139159mm，score/flag不变，仍漏报199/200。
K2对照：ID RMSE.125224mm、score中位数.034299mm、threshold=.114464mm；
state/action高误差分别34/113，漏报1/6；hidden200条高误差漏报197条。
K2与K5的前两个成员权重/bootstrap完全一致，改变K使均值、分歧与阈值重新计算。

首轮batch1完整5成员推理、mean和score p50/p95=.033752/.066264ms，20warmup/200samples；
不含训练、truth、I/O、环境或planning，不是机器人端到端latency。
固定种子单次ensemble结果不是多次独立训练的置信区间。

## Explanation / Result

本ensemble的score在两类指定输入偏移上表现出一定诊断价值，但对不可见机制变化无响应。
这既提供实际learned模型反例，也用复制成员说明“成员数多”不能取代独立信息。
未学习概率校准、未估计aleatoric噪声，也未证明能可靠检测任意OOD。
没有policy/闭环任务，robot success、失败stage和控制latency均未测。

## Failure Analysis / Failure Cases

| 问题 | 处理 / 限制 |
|---|---|
| 相邻帧bootstrap或split泄漏 | split在episode层，bootstrap只采train IDs，不采calibration/test |
| normalizer或threshold偷看OOD | 统计只train，阈值只独立名义calibration，不据OOD结果调参 |
| 成员共享偏差、输入缺机制 | hidden shift 199/200漏报；小score不保证准确 |
| 复制同一成员冒充ensemble | 负对照score0且全部漏报，不当作新学习成员 |
| score当概率或误差界 | 单位m、无coverage guarantee，不报告confidence百分比 |
| OOD高flag即成功检测失败 | state shift有151条诊断FP，report完整分母 |
| 给同分布标签等同每点ID | ID是生成分布关系，有限train范围不能定义全部分布支持 |
| 把OOD tuple当trajectory | 显式标记单步反事实，不评价多步不确定性传播 |
| 均值掩盖个别模型/样本 | report逐成员误差，NPZ/CSV保留所有样本，不删失败 |
| 换seed/K就一定改善 | 无单调保证；共享架构仍相关，本次非多seed总体统计 |

## Relation to Robotics

分歧可成为后续动作审核或采集更多数据的输入，但应先验证它与实际任务失败的关系。
单位m的预测分歧不能直接当碰撞概率，低分歧也不能授权执行真实动作。
本任务不提前实现S3.5风险head、S3.6规划或Stage4动作拒绝。

## Interview Capsule / Interview Notes

30秒：我训练5个226参数动力学MLP，用episode bootstrap和独立初始化增加多样性。
在物理next-state上算成员population RMS分歧，阈值只来自独立名义calibration。
ID RMSE约.125mm；指定输入OOD时误差/分歧增大，但同输入隐藏执行变化会使199/200高误差漏报。
分歧是诊断分数，不是校准风险，也没有证明机器人闭环成功。

2分钟：补充K×N×2 shape、2K分母、train-only统计、机制变化与输入变化、
独立calibration和evaluation、TP/FP/FN/TN阈值语义、复制成员与共偏差的不变性。
区分epistemic proxy、aleatoric noise与校准failure probability，最后交代CPU成本和toy证据边界。

## Likely Follow-up Questions

1. Ensemble有哪些多样性来源？为什么按episode bootstrap、共享统计又只fit原train，而不把validation/calibration用于训练？
2. 写出(K,N,2)预测的均值和D公式，解释分母、单位以及它和L2/RMSE/standard error的差别。
3. q95来自哪组数据？flag与L2>5mm标签分别是什么？为何不能声称ID误报率固定5%或flag是失败概率？
4. shift10→20mm时哪些产物不变？为什么hidden shift和复制成员会低分歧但高误差？
5. 本包怎样区分epistemic proxy、aleatoric与mechanism shift？输入OOD、成员更多或低分歧分别不能推出什么？

## Must Remember

成员不同不等于错误独立；disagreement≠error≠calibrated risk；train-only统计；独立阈值集；同输入机制变化可完全漏检。

## My Verification / Learner Verification

- [x] Run：运行默认实验，核对四组误差/分歧与完整混淆矩阵。
- [x] Modify：先预测shift10→20mm的不变量，再运行核对；可选K2对照。
- [x] Explain：用自己的话回答五问，解释低分歧高误差与阈值证据边界。

Engineering Complete + Learning Mastered（2026-10-20 用户明确确认实验和预测完成，并准确回答五项Explain）。
解释覆盖episode bootstrap与独立初始化、train-only共享统计、K×N×2与2K分母、
独立q95阈值和flag/bad语义、隐藏机制/复制成员反例，以及epistemic/aleatoric与风险校准边界。
Current Task=S3.3；Next Task=S3.4仅计划。
STOP，不自动开始下一任务。
