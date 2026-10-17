# S1.1 — Language-conditioned State Policy

## Problem

同一个当前位置，`move left` 与 `move right` 应得到不同动作。S0.7是手写planner；
本任务第一次训练参数，比较语言条件MLP和同规模无语言MLP。范围止于state policy，H=E=1。

## Intuition

当前位置告诉模型“在哪里”，指令告诉模型“去哪个目标”。没有指令时，同一个位置会对应冲突的标签；
MSE容易学成折中，或沿数据里的位置相关性行动。比较两种模型才能判断语言是否有用。

## Input / Output Contract

30个独立reset，固定seed11采样初始world XY：x∈[−.18,.18]m，y∈[−.15,.15]m。
每组左右指令各一episode，相同起点；先按group划分：0–23 train，24–29 validation。
每episode10次动作、11次观测，dt=.1s；60episodes，train480/validation120训练样本。
目标固定left[−.24,0]、right[.24,0]m。理想点执行p_next=p+a，所有动作预算结束truncated，success独立。

| 字段 | shape / dtype | 含义 |
|---|---|---|
| raw state | (11,2) float32 | t=0…10 measured world XY，m |
| raw action | (10,2) float32 | t→t+1 executed world displacement，m，每分量限幅±.06 |
| raw clock | (11,) float64 | 相对秒，0…1 |
| language | JSON string | episode常量，只接受两条literal指令 |
| input x | (N,4) float32 | [x/.3,y/.3,left_bit,right_bit]，当时state，不含future/success/group |
| target | (N,2) float32 | raw action保留m；内部训练除以.06，变无量纲 |
| hidden | (N,32) float32 | tanh特征 |
| output | (N,2) float32 | 线性输出×.06恢复m；执行前clip到±.06 |

位置/.3、动作/.06是预先声明的常量尺度，未从validation拟合统计。没有图像、history或padding。
复用embodied_episode/1.0和S0.1 validator，raw NPZ+JSON先round-trip再取state_t/action_t。
不同于S0.1，这里的left/right表示趋近固定目标，不是永远向左/右；经过目标后正确动作会反向纠偏。
unknown literal、非float32/非有限state拒绝，未实现自然语言encoder。

## Concepts

Behavior cloning用expert动作监督；参数由train loss更新。Expert是手写标签生成器，policy是学习得到的MLP。
无语言基线重新训练同结构网络，仅把两个语言位设0，不是把训练后的模型临时破坏。
同起点的左右episode保留在同group，不能拆到不同split。held-out只是同分布新reset，目标和语言都已见过。
固定训练步数，不用validation选择模型；本实验没有独立test set或多seed统计置信区间。

## Mathematics

专家：a*=clip(g(ℓ)−p,−.06,.06)。输入x=[p/.3,one_hot(ℓ)]。

h=tanh(xW1+b1)，u=hW2+b2，a_raw=.06u，a_exec=clip(a_raw,−.06,.06)。
W1:(4,32)，b1:(32,)，W2:(32,2)，b2:(2,)，共226个参数。
L=mean((u−a*/.06)²)，无量纲；物理MSE=L×.06²，单位m²。

手写反向传播：D=2(u−y)/y.size；dW2=hᵀD；db2=sum(D)；
B=(DW2ᵀ)⊙(1−h²)；dW1=xᵀB；db1=sum(B)。
Adam：m=.9m+.1g，v=.999v+.001g²；偏差修正后W←W−.01 m_hat/(sqrt(v_hat)+1e−8)。
全batch、默认2000次更新、初始化seed7；有限差分在float64小网络上逐参数核对梯度。

闭环每.1s重新取真实state，运行MLP并执行一个delta，固定10步；
最终max|p−goal|≤.01m算成功。保持完整分母，记录失败stage和trace，不把低offline MSE当成功。

## Architecture / Data Flow

reset + literal instruction → expert raw episodes → reset-group split → current state + one-hot
→ fixed scaling → Linear(4,32) → tanh → Linear(32,2) → physical delta → clip → ideal point → fresh state。
labels只进入loss；future state、success、group和clock不进入policy。

## Math / Architecture → Code

[script](../../scripts/s1_1_language_state_policy.py)：`dataset`生成并验证raw、按固定group取样；
`encode`实现显式输入契约；`forward/gradients/train`对应MLP/反向/Adam；
`predict`解码物理delta；`evaluate`保留全部闭环轨迹、动作、失败与延迟。
`gradient_check`验证手写导数；权重NPZ重新载入后逐值比较推理结果。

## Experiment

仓库根目录，现有NumPy即可，无安装、数据或weights下载。输出必须新目录：

```bash
.venv/bin/python scripts/s1_1_language_state_policy.py --output tmp/my_s1_1
# 先预测减少训练会影响哪些量，再运行：
.venv/bin/python scripts/s1_1_language_state_policy.py --epochs 20 --output tmp/my_s1_1_short
```

查看report.json中history、两模型metrics、same_state_actions_m和rollout；
查看train/validation.npz和episodes/*/episode.json。NPZ禁止pickle，所有生成物在ignored tmp中。

## Expected vs Actual Result

预期：含语言模型能学会在同state下按指令分叉；无语言模型无法区分完全相同输入的冲突目标。
减少训练不改变数据、split或参数量，可能增大误差与失败率；不保证每一步loss或success严格单调。

2026-10-17 CPU实测，repo .venv Python3.14.7、NumPy2.5.4；actual module在venv中。
本环境与旧handoff的Python3.12不同，未来Torch仍需重新核验；本任务没有Torch依赖。

| 模型 / epochs | train raw MSE[m²] | validation raw MSE[m²] | held-out toy success |
|---|---:|---:|---:|
| conditioned / 2000 | 1.048e−6 | 1.349e−6 | 12/12 |
| no_language / 2000 | 4.283e−4 | 4.669e−4 | 6/12 |
| conditioned / 20 | 2.884e−4 | 2.850e−4 | 3/12 |
| no_language / 20 | 6.753e−4 | 6.839e−4 | 1/12 |

默认同state[0,0]，左右输出x≈−.06/+ .06m；无语言两次均≈+.003695m。
无语言基线并非必然完全不动，后续state可被数据相关性引向某一目标，但无法按指令选择。
report另存执行限幅后的MSE，避免掩盖raw超限误差。所有失败stage均为10步预算后的终点误差。
梯度最大绝对差≈1.40e−11；raw契约/转移/终点、round-trip、reset不重叠、模型reload通过。
延迟逐模型记录120次encode+MLP+clip的median/p95，排除环境；本机调度相关，不是部署性能结论。

## Result / Explanation

证据支持这个小MLP利用literal条件解决固定两目标的toy任务。2000步和20步的对比说明
“同state不同输出”只证明条件响应，还需检查动作精度和闭环累积误差。当前未证明真实语言理解或机器人泛化。

## Failure Analysis / Limitations

- 未见指令直接拒绝；不能推断paraphrase或未知词能力，语言表示留S1.2。
- 固定目标、无视觉、无接触/惯性/噪声；不能称Tiny VLA或foundation model。
- 离线分布包含expert到达后的大量zero动作，平均MSE可能掩盖早期错误；闭环保留全轨迹核查。
- no-language虽然在6/12成功，也不能据此称会理解指令；同state输出相同。
- validation只6个新reset，同目标/同指令；多次查看结果不是独立test证明。
- 预测误差会改变未来输入，20步模型有终点失败；没有不确定性校准或安全保证。

## Relation to Robotics

输入是proprioception和任务条件，输出是明确world delta。理想加法控制器不能替代真实servo/IK；
接真实数据前仍须审计action frame、anchor、controller和时间，不导入旧repo模块。

## Interview Notes / Capsule

30秒：我用226参数NumPy MLP把world XY和两类one-hot映射成delta，手写反向并有限差分验证。
按reset划分，含语言/无语言分别训练；held-out toy闭环12/12与6/12，不能推广为自然语言理解。

2分钟：补充输入/输出尺度、MSE物理单位、train-only参数更新、无语言冲突标签、raw与限幅误差区别，
以及20步模型虽然已条件分叉但仅3/12成功，解释offline loss与闭环偏移的差别。

## Likely Follow-up Questions

1. 输入4维各代表什么？raw/训练/执行动作分别是什么单位、shape、时间和frame？
2. one-hot是语言理解吗？为什么相同state不同输出只能算必要检查？
3. 为什么按reset分组？为何不能把左右pair拆开？固定尺度与fit normalizer有何区别？
4. 无语言基线为何可能成功6/12？为什么不能要求它总输出0？
5. 2000→20步哪些东西不变？为什么需要闭环、失败stage和延迟，而不只报告loss？

## Must Remember

条件响应≠语言泛化；offline loss≠闭环成功；expert规则≠learned policy；原始物理数据不覆盖。

## My Verification

- [ ] Run：运行默认实验，读两模型metrics、同state动作和12条rollout。
- [ ] Modify：预测2000→20步对数据、参数量、loss、同state动作、success的影响，再运行核对。
- [ ] Explain：独立回答上述五问。

Current Task = S1.1；Next Task = S1.2（仅计划）。工程完成，学习等待用户报告，不自动推进。
