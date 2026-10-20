# S3.1 — State Dynamics

## Problem / Why

VLA 回答“现在做什么动作”；动力学模型回答“给定当前状态和动作，下一状态是什么”。
本任务学习 `(s_t,a_t) → s_{t+1}` 的小 MLP，与不动/理想加法基线比较。
仅单步、完整状态、CPU；不实现多步 rollout、uncertainty、planning 或 VLA 集成。
预计学习 1.5 小时：概念与公式20分钟，源码20分钟，运行修改20分钟，解释与失败分析30分钟。

## Intuition

发出“向右移动 1cm”的命令，不代表实际测量一定向右移动 1cm。
这里人为设置位置相关执行增益和 XY 串扰，模型需要学会命令与实际结果的差别。
这些是合成规律，不是辨识真实机器人的证据。

## Input / Output Contract

独立 profile `s3_1_xy_command_dynamics/1.0`，每次输出 `contract.json`；
不是 S0.1 ideal delta profile，也不将它伪装成旧数据的同语义版本。

| 项 | shape / dtype | 单位、frame、时序 |
|---|---|---|
| raw state | (40,21,2) float32 | world XY位置，m；每episode一次独立reset |
| raw action | (40,20,2) float32 | world XY **commanded displacement**，m，作用于t→t+1 |
| timestamp | (21,) float64 | 相对秒，dt=0.1s；所有episode同一时间网格 |
| train input | (600,4) float32→float64 | `[x_t,y_t,a_x,t,a_y,t]`，独立逐通道标准化 |
| train target | (600,2) float32→float64 | 下一位置减当前位置，标准化后的实际位移 |
| validation | input (200,4), target (200,2) | 10条从未进入训练或统计拟合的episode |
| MLP output | (B,2) float64 | 标准化位移→反标准化[m]→加当前位置得下一位置 |

40个episode各20次转移，固定时限截断，无success标签；没有视觉、语言、速度或接触。
动作不是绝对目标、速度，也不是已实现位移。不从 sibling repo 读取或导入数据。
raw NPZ保留物理值；下一状态只作为监督目标，不拼入模型输入。
初态每轴均匀采样[-0.3,0.3]m，动作每轴均匀采样[-0.04,0.04]m；
后续状态由动力学产生，未强制截断到初态范围。随机种子data/split/model=31/32/33。

## Core Concepts

- Forward dynamics：状态和动作预测结果；inverse dynamics与policy的输入输出不同。
- Markov state：本toy的下一状态只依赖当前位置和动作。真实有惯性系统往往还需要速度。
- Residual prediction：网络学实际位移，不是直接回归大尺度绝对位置，也不是将命令当位移。
- Episode split：先把独立reset episode分开，再展平转移；同episode相邻帧不能跨split。
- 单步验证使用真实当前状态；不能据此断言模型自行连续预测时稳定。

## Mathematics

逐轴执行，`reverse_xy`交换动作的两个分量：

\[
g(s)=0.70+0.15\tanh(s/(0.25\,\mathrm m)),\qquad
s_{t+1}=s_t+g(s_t)\odot a_t+0.08\operatorname{reverse}_{xy}(a_t).
\]

s和a都是world XY的2维m向量；g、0.08无量纲。这是人为确定性转移，无噪声或隐藏状态。
理想加法 `s+a` 忽略了增益和串扰；persistence预测`s`，忽略动作。
即使不动基线绝对位置误差很小，也要检查它是否只利用dt短、运动小。

\[
x=[s_t,a_t],\quad y=s_{t+1}-s_t,\quad
\tilde x=(x-\mu_x)/\sigma_x,\quad \tilde y=(y-\mu_y)/\sigma_y,
\]
\[
h=\tanh(\tilde xW_1+b_1),\quad \hat{\tilde y}=hW_2+b_2,\quad
\hat s_{t+1}=s_t+\mu_y+\sigma_y\odot\hat{\tilde y}.
\]

所有mean/std只fit 600条train转移。训练loss为标准化位移的元素MSE，单位无量纲；
评价MSE=`mean((s_hat-s_next)^2)`单位m²，RMSE单位m，是每坐标RMSE；
p95 L2是200个转移的二维欧氏误差95分位，不能与逐坐标RMSE直接等同。

反向：`d=2*(pred-y)/y.size`，`dh=(d@W2.T)*(1-h*h)`；
`dW2=h.T@d`，`dW1=x.T@dh`，bias沿batch求和。全batch Adam学习率.01，β=.9/.999，ε=1e-8。
226参数：4×32+32+32×2+2。不是foundation/world foundation model复现。

## Architecture / Data Flow

`synthetic reset → execute commands → 21 states / 20 actions → episode split → flatten`

`train-only statistics → [state,action] → normalize → Linear(4,32) → tanh → Linear(32,2)`

`→ inverse normalize delta → add current state → next-state prediction → physical held-out error`

模型输入不包含transition函数、gain或未来状态。generator用于生成标签；evaluation不调用generator为模型修正预测。
真实传感器/执行器延迟、估计误差和controller都没有模拟。

## Math / Architecture → Code / Implementation

[脚本](../../scripts/s3_1_state_dynamics.py)：`transition`定义合成物理规律，
`dataset`保存reset内时序，`forward`和`gradients`显式实现两层网络与反向，
`main`先split再拟合统计并训练，`metrics`恢复物理单位后评价三个预测器。
训练前对四个参数tensor各取一个最大梯度坐标，中心差分ε=1e-5核验；不是穷尽梯度证明。
`model.npz`保存权重与统计，`episodes.npz`保存未归一化状态/动作、时钟与split索引；
`validation_predictions.npz`供逐条审计，读取用`np.load(..., allow_pickle=False)`。

## Minimal Experiment

无需安装或下载。已验证repo隔离.venv Python3.12.3 / NumPy2.5.4；模型计算float64，CPU。
从仓库根目录执行；输出必须为新目录。限制BLAS线程降低小矩阵的调度开销：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_1_state_dynamics.py --output tmp/my_s3_1
# 先预测减少训练步数对数据、统计、基线和MLP误差的不同影响，再运行：
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_1_state_dynamics.py --epochs 20 --output tmp/my_s3_1_short
```

查看contract.json、report.json和validation_predictions.npz。
2000/20步使用完全相同的数据、split、初始化和统计，只改变优化步数。
验证集只用于最终报告，不做早停或选择最佳checkpoint；这些固定seed结果不是多seed统计。

## Expected Result

2000步MLP应明显优于两个基线；20步误差预计更大，数据/split/基线保持不变。
训练loss可能局部振荡，不保证每步单调或任意seed同样结果。

## Actual Result

2026-10-20 CPU，Python3.12.3 / NumPy2.5.4，单BLAS线程，train600 / validation200转移：

| Predictor | held-out MSE [m²] | held-out RMSE [mm] | held-out p95 L2 [mm] |
|---|---:|---:|---:|
| 不动 | 2.98838e-4 | 17.287 | 37.144 |
| 理想加法 | 4.69025e-5 | 6.849 | 16.484 |
| MLP 2000步 | 1.90081e-8 | 0.138 | 0.300 |
| MLP 20步 | 9.58777e-6 | 3.096 | 7.378 |

默认train RMSE约0.099mm，最终标准化train loss约3.78992e-5；
四个梯度探针最大绝对误差2.692e-11。
首轮2000步batch1单步推理p50/p95约0.00594/0.00745ms（30 warmup、200样本）。
计时含标准化、网络、反标准化和加法；不含观测、I/O、规划或机器人执行，重跑有波动。

## Explanation / Result

该MLP在同分布独立episode的单步预测中优于忽略命令或忽略执行规律的基线。
未证明最优架构、真实机器人动力学辨识、OOD泛化、接触建模或部署成功。
没有policy或机器人任务，因此closed-loop success、控制失败stage均未测；不虚构成功率。

## Failure Analysis / Failure Cases

| 问题 | 后果 / 当前边界 |
|---|---|
| command当realized delta | `s+a`产生系统偏差；本toy显式gain/串扰 |
| 仅位置用于惯性机器人 | 非Markov输入，同一位置动作可有不同下一状态；本toy无惯性 |
| 相邻帧随机split | trajectory相似性泄漏；当前以完整独立reset split |
| 全数据normalization | 验证分布进入训练；当前统计只来自train |
| 训练不足或优化振荡 | 20步误差更高；完整保存loss，不挑验证最好结果 |
| 范围外动作/状态或换dt | 未评价OOD，不作外推声明 |
| 连续预测积累误差 | 单步误差不能证明长时稳定；S3.2才授权研究 |
| 随机性/接触/遮挡 | 当前完全未覆盖；确定性MSE模型不提供可信风险概率 |

## Relation to Robotics

动力学模型可以成为规划的预测器，但还需要任务cost、候选动作、约束与环境验证。
学得准确的局部单步模型不自动产生好policy，也不能替代低层控制器。

## Interview Capsule / Interview Notes

30秒：我用当前world XY位置和位移命令，预测执行后的下一位置。合成系统有位置相关增益和串扰，
因此命令不等于实际位移。226参数MLP学实际位移，按完整episode划分，统计只fit train；
held-out单步RMSE约0.138mm，理想加法约6.85mm。它没有证明多步稳定或机器人成功。

2分钟：补充4→32→2结构、残差反标准化、无量纲训练loss与m²评价区别；
解释T+1观测、Markov假设、基线为何必要，以及真实机器人速度/controller/dt/frame如何改变契约。
最后指出还需要多步/OOD/接触、闭环任务成功和端到端latency证据，不能用离线loss替代。

## Likely Follow-up Questions

1. VLA policy与forward dynamics的输入输出分别是什么？为什么这次不需要language？
2. action的单位是m，为什么仍不能直接`s_next=s+a`？真实有惯性系统为何位置通常不够？
3. 画出20 actions / 21 states时序；解释4维输入、2维target、残差恢复下一状态。
4. 为什么先episode split再flatten？mean/std可以用validation吗？train loss与报告RMSE单位是什么？
5. epochs 2000→20时哪些artifact/基线应不变？单步更准为什么仍不能推出控制成功？

## Must Remember

命令≠实现位移；forward model≠policy；train-only统计；episode split；单步误差≠多步稳定≠robot success。

## My Verification / Learner Verification

- [x] Run：自己运行默认命令，查contract、数据时序和三个held-out误差。
- [x] Modify：先预测2000→20步的变化，再运行比较数据/统计/基线和MLP。
- [x] Explain：用自己的话回答五问，讲清action语义、Markov和评价边界。

Engineering Complete + Learning Mastered（2026-10-20 用户明确确认实验和预测完成，并准确回答五项Explain）。
解释覆盖policy/dynamics输入输出、commanded/realized位移、Markov状态、20/21时序、残差恢复、
episode split、train-only统计、误差单位、训练步数不变量及单步/闭环证据边界。
Current Task=S3.1；Next Task=S3.2（仅计划）。STOP，不自动进入下一任务。
