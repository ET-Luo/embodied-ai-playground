# S3.4 — Latent State Representation

## Problem / Why

S3.1–S3.3直接使用world XY状态。若输入是一组观测通道，我们能否先压缩成z，
再用 `(z_t,a_t)→z_{t+1}` 预测？本包分别测量重建、latent预测、解码后的物理预测，
避免把低latent loss当作保留了机器人任务需要的信息。
预计2小时：Problem→Mathematics30分钟，源码25分钟，Run/Modify25分钟，Explain/失败分析40分钟。
只实现线性数值观测encoder与小latent MLP，不做视觉网络、大视频模型、风险head或规划。

## Intuition

六把朝向不同的“尺子”同时测量同一个二维位置，这六个读数是冗余的。
把它们压到两个坐标，可以保留全部位置；只留一个坐标，则不同位置可能得到相同编码。
如果动力学依赖被丢掉的位置分量，相同z和相同动作也会产生不同的下一z。
重建观测和预测未来是两种目标；对重建有用的主方向不自动是控制或动力学最需要的方向。

## Input / Output Contract

profile `s3_4_linear_sensor_latent/1.0`，raw物理规律沿用
[S3.1](s3_1_state_dynamics.md) 的位置相关gain/XY串扰。观测是新建的合成数值传感器，
不是旧数据的静默重解释；每次输出contract.json。

| 项 | shape / dtype | frame、单位与时序 |
|---|---|---|
| raw state | (40,21,2) float32 | world XY位置[m]，用于generator/评价，不输入模型 |
| raw observation | (40,21,6) float32 | 六路已知world XY线性投影长度[m]，无噪声 |
| raw action | (40,20,2) float32 | commanded world XY displacement[m]，作用于t→t+1 |
| timestamp | (21,) float64 | 相对秒，dt=.1s，无terminal action |
| encoder fit batch | (630,6) float64 | 30条train episode所有21个状态，每个恰好一次 |
| PCA axes C | (6,d) float64 | d=2默认，d=1对照；列正交，符号规范化仅为可重复 |
| encoded z | (episode,21,d) float64 | 无量纲；学习坐标，没有直接m单位或world轴含义 |
| dynamics input | (600,d+2) float64 | train当前z+物理命令，逐通道train标准化 |
| dynamics target | (600,d) float64 | delta_z=z[t+1]−z[t]，train均值/尺度标准化 |
| predicted z_next | (N,d) float64 | 当前z+反标准化的网络delta_z |
| decoded observation | (N,6) float64 | 反PCA/反观测标准化后的六路长度[m] |

40条独立reset，每条20次转移、21个观测，最后time truncation，无success标签。
按S3.1 seed32完整episode划分train30/validation10，数据seed31，动力学初始化seed340。
train统计与PCA使用630个独立时间槽，包含train terminal observation；
dynamics输入/delta统计使用600个转移，不能把terminal加成第601个action样本。
离线fit train encoder允许使用train终点；推理只编码当时观测，没有future observation输入。
validation的210观测/200转移不参与PCA、任何mean/std或模型梯度。
没有image、language、force、velocity、接触或随机过程；PCA不是视觉encoder复现。

### 显式测量模型

按行列出6×2矩阵A，`o=s @ A.T`，每个通道为固定无量纲权重乘world XY长度：

```text
A = [[ 1.0,  0.3],
     [-0.4,  1.0],
     [ 0.8,  0.6],
     [ 0.6, -0.8],
     [-0.8,  0.6],
     [ 0.3,  1.0]]
```

不是RGB，也不是现实相机或接触测量。A已知且满列秩；物理评价使用
`decoded_observation @ pinv(A.T)`恢复XY。这是特权synthetic测量几何，
不是模型直接读取真实XY，也不是学到了未知机器人的sensor calibration。
raw保留float32物理值，SVD/模型/解码float64；极小重建残差包含float32观测舍入误差。

## Core Concepts

- Encoder E把观测压成z，decoder D尝试还原观测；dynamics F用z和动作预测下一z。
- PCA是**用训练数据拟合的线性encoder/decoder**，不是手写挑两个通道，也不是神经autoencoder。
- 两阶段：先拟合PCA并冻结，再训练latent dynamics；目标E(o_next)保持固定，不联合优化。
- 表示坐标可旋转、变号或缩放；raw latent MSE依赖坐标约定，不能跨不同encoder直接当物理精度排名。
- 重建保留大方差不等于预测所需信息充分。压掉信息会产生aliasing，甚至破坏latent Markov性质。
- 零/常量encoder可有很小的latent预测loss，但不能重建物理状态；因此同时检查方差、重建和解码预测。
  本包PCA1不是完全collapse，它仍保留一个主方向，但丢失另一个物理方向。

## Mathematics

行向量写法，观测通道数6；所有观测统计只fit train630个时间槽：

\[
\tilde o=(o-\mu_o)/\sigma_o,\qquad
\tilde O=U\Sigma V^T,\quad C=V_{:,1:d},\quad C^TC=I_d.
\]

\[
z=E(o)=\tilde oC,\qquad D(z)=(zC^T)\odot\sigma_o+\mu_o.
\]

SVD按奇异值大小给主轴；取前d个方向，是当前标准化训练矩阵的最小平方线性低秩重建。
标准化后优化的是每通道标准化残差，不是直接最小化world XY误差。
`sum(singular[:d]²)/sum(singular²)`是保留的训练标准化观测方差比例，不能叫任务成功率。
这里六路无噪声线性观测来自两个物理自由度，d2几乎可无损重建；真实观测不具备这个强假设。

\[
x=[z_t,a_t],\quad y=z_{t+1}-z_t,\quad
\hat z_{t+1}=z_t+\mu_y+\sigma_y\odot
\operatorname{MLP}_{\theta}((x-\mu_x)/\sigma_x).
\]

MLP为(d+2)→32 tanh→d，d2有226参数，d1有161参数；
mean/std只fit train600，loss为标准化delta_z的元素MSE，无量纲。
手写反向复用S3.1通用两层tanh梯度；full-batch Adam lr=.01、β=.9/.999、ε=1e-8，2000步。
训练前四个参数tensor各挑一个活跃坐标做中心差分ε=1e-5，非穷尽梯度证明。

评价同时看：

1. 重建：`D(E(o))` 对 o；再经已知测量逆映射，对真实s。
2. 单步latent误差：F(E(o_t),a_t) 对 E(o_next)，无量纲、未标准化z坐标中的MSE。
3. 单步decoded误差：D(F(E(o_t),a_t)) 对 o_next，及物理XY误差。
4. Future-encoding重建参考：D(E(o_next))，用于观察encoder/decoder本身的信息损失；
   它使用未来真值，是**特权诊断**，不是部署预测，也不保证是物理误差的严格下界。
5. 20步AR：仅E(o0)初始化，递归F并解码，评价prefix与terminal，期间不重新编码未来真实观测。

六通道重建MSE分母N×6，物理XY MSE分母N×2；单位分别为m²，RMSE单位m。
latent MSE分母N×d；不能与物理MSE数值直接比较。
validation重建包含210观测，单步预测包含200转移；AR20端点只10个末状态，prefix包含200个预测点，不含t0。

## Architecture / Data Flow

`world XY generator → six sensor projections → episode split`

`train630 observations → mean/std → SVD/PCA E,D → freeze`

`train600 (E(o_t),action_t,E(o_next)) → train-only x/delta statistics → latent residual MLP`

`validation o_t → E → z_t + action → F → predicted z_next → D → predicted observation`

`→ known sensor inverse → physical XY metrics`

future observation只用于监督target和特权重建参考，不进入预测分支。
state XY仅用于生成观测、计算标签/评价和特权ideal-addition基线。

## Math / Architecture → Code / Implementation

[脚本](../../scripts/s3_4_latent_state.py)：`observe`实现A，`fit_encoder`fit train统计并SVD，
`encode/decode`对应PCA公式，`fit_dynamics`只读取train z与动作；
`predict_latent`构造residual预测，`rollout`只接受起始z和预定动作，`evaluate`区分各类误差。
`physical_state`为固定几何逆，不是learned head。

`model.npz`保存观测统计、PCA axes/奇异值、latent dynamics统计和权重、sensor_matrix；
`episodes.npz`保存raw state/observation/action/clock及split；
`validation_predictions.npz`保存真实latent、重建、单步预测/监督目标与完整20步AR，
另有report.json与contract.json。读取用`np.load(..., allow_pickle=False)`。

[测试](../../tests/test_s3_4_latent_state.py)覆盖d2 held-out重建、d1物理aliasing、
改future observation不影响AR与h1一致性、无效/退化观测拒绝。

## Minimal Experiment

repo .venv Python3.12.3 / NumPy2.5.4，CPU，无新依赖、安装、数据或weights下载。
自包含生成并训练，不改S3.1–S3.3 checkpoint。目录必须不存在：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_4_latent_state.py --output tmp/my_s3_4
# 先预测d2→d1对raw数据、PCA统计、重建、latent/物理误差与参数shape的影响：
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_4_latent_state.py --latent-dim 1 --output tmp/my_s3_4_d1
```

先读contract，再看report validation的reconstruction、decoded_next、future_encoding_reconstruction_reference，
以及20步AR。`zero_action_ablation_physical`将推理时物理命令置0，再正确标准化，检查动作信息影响。
`latent_persistence`直接预测z_next=z_t；`privileged_ideal_addition`用真实state+命令，
是不同输入信息的物理参照，不能当同输入同预算模型排名。

## Expected Result

d2应近乎无损重建，latent dynamics应比latent persistence单步更准；去动作预计变差。
d1应保留较少观测方差并出现明显重建与物理预测损失；latent误差小也未必反映物理信息充分。
d2→d1时raw数据/split/观测mean/std/全部奇异值保持不变，PCA axes由6×2变6×1，
后续latent统计、模型输入输出shape、权重与误差改变。不同d不能只比较raw latent MSE。

## Actual Result

2026-10-20，CPU单BLAS线程，分别训练2000步：

| validation指标 | d2 | d1 |
|---|---:|---:|
| 保留train标准化观测方差 | ≈100% | 55.6746% |
| observation重建RMSE [mm] | 3.282e-6 | 116.6687 |
| physical XY重建RMSE [mm] | 2.360e-6 | 121.2081 |
| latent预测MSE（无量纲） | 1.09065e-6 | 7.22123e-5 |
| latent persistence MSE（无量纲） | .0247726 | .0247894 |
| decoded next XY RMSE [mm] | .116069 | 121.9010 |
| future encoding重建参考 XY RMSE [mm] | 2.376e-6 | 121.8943 |
| latent persistence decoded XY RMSE [mm] | 17.2869 | 122.4230 |
| zero-action消融 decoded XY RMSE [mm] | 17.2852 | 122.4185 |
| AR20末状态 XY RMSE [mm] | .758313 | 154.7802 |

两个d的latent模型都显著降低自身latent persistence误差，但d1几乎不能改善物理精度，
其物理误差主要伴随表示丢失。不可把d1小latent MSE解释为state可预测或控制可用。
d2 train/validation decoded next RMSE=.085263/.116069mm，仅相同合成生成分布留出证据。
四tensor梯度探针最大误差d2约1.985e-11、d1约4.324e-11。

**d1 aliasing探针**：沿encoder物理null方向构造相距.1m的两个位置，它们编码逐值差<1e-12；
执行相同[.03,.02]m命令后，真实下一latent差约.0016663。
说明F(z,a)面对相同输入却有不同真实target，无法对两者同时精确；这是表示信息不足，不仅是优化不足。
探针使用float64线性观测做严格代数相等性，真值transition/后续观测仍按raw float32约定。

首轮d2 batch1完整encode→predict→decode p50/p95=.009404/.010985ms，20warmup/200samples；
不含sensor生成/传输、真值、physical evaluation、I/O或planning，非机器人闭环latency。
后续最终版仅清理未使用代码、将重建诊断命名为reference，数值模型结果一致。
13项Stage3协议测试通过；checkpoint相同计算路径单步/AR逐值重载一致。
批量单步与逐步AR的h1比较使用atol1e-14/rtol0，d1实际最大舍入差仅2.776e-17（无量纲）。

## Explanation / Result

本toy里d2表示能保留位置且支持动作条件预测；d1揭示重建压缩和动力学充分性之间的冲突。
PCA只优化观测重建，未来复杂视觉任务可能需要其他表示目标，但本轮没有实现或评价它们。
已评价encoder/decoder和dynamics在held-out数据上的行为，不证明真实视觉编码、OOD泛化、
latent uncertainty、接触建模或长时稳定。没有policy或机器人任务，不报告虚构success/failure stage。

## Failure Analysis / Failure Cases

| 问题 | 处理 / 当前限制 |
|---|---|
| PCA先fit全部数据再split | 先episode split，encoder只train630；validation不进统计 |
| 为fit encoder重复计入相邻target | 对每条原始train episode的21观测各取一次，非转移两端重复拼接 |
| encoder在dynamics训练中变化 | 冻结E/D，避免moving target；未实现joint loss或神经AE |
| latent丢失关键状态 | d1 aliasing+大physical误差，低latent loss无法弥补 |
| latent维数/坐标任意却比较raw loss | 同时报告物理误差、重建与明确坐标/分母，无跨encoderloss排名 |
| 将未来编码输入AR | 反事实改future observations，AR不变；特权reference单独命名 |
| 重建参考当严格误差下界 | 它是诊断参照，物理度量下不保证最优或lower bound |
| 知道A就声称解决视觉 | 仅已知线性数值sensor，无图像、遮挡或sensor calibration学习 |
| 维数2就一定Markov | 本toy满秩线性测量+位置Markov；真实惯性/隐藏接触仍需额外状态/历史 |
| 单步/20步准就可部署 | 未测试真实系统、任务成功或planning，保留完整失败案例 |

## Relation to Robotics

latent可以是world model的内部状态，但既要保留观测信息，也要保留决定未来与任务结果的信息。
即使重建漂亮，也可能漏掉速度、接触或小目标；本toy只用可计算的aliasing展示这个问题，
没有把它推广成真实sensor representation的实验结论。

## Interview Capsule / Interview Notes

30秒：我把world XY转换为六路已知线性数值观测，只用train630个观测拟合PCA encoder/decoder，
冻结后训练小MLP从z和命令预测下一z。d2验证单步物理RMSE约.116mm；d1虽显著优于latent persistence，
物理误差仍约122mm。存在相同z/动作却不同下一z的反例，因此低latent loss不代表动力学充分。

2分钟：补充630观测vs600转移、train-only统计、SVD方向与尺度、残差预测和解码测量逆，
分开重建/未来编码参考/预测误差与AR端点分母；最后说明线性无噪声toy与真实视觉/闭环差别。

## Likely Follow-up Questions

1. E、D、F各自接收/输出什么shape和单位？这里的PCA encoder与神经autoencoder有什么区别？
2. 为什么encoder fit630条观测，而dynamics fit600条转移？train terminal用于fit为何不是推理时未来泄漏？
3. 重建误差、latent预测误差、decoded物理误差及future-encoding参考分别测什么？为什么不能只看latent loss或把reference当部署结果？
4. d2→d1哪些raw/statistics保持不变、哪些shape和结果改变？相同z/动作但不同下一z说明什么？
5. AR20中如何防止未来观测泄漏？d2近乎无损重建与低物理误差为什么仍不证明视觉理解、真实系统Markov或机器人成功？

## Must Remember

latent不是物理单位；重建≠预测≠控制；train-only encoder；未来编码只作target；信息丢失可破坏动力学充分性。

## My Verification / Learner Verification

- [x] Run：运行d2，查看contract、重建/预测/AR与动作消融。
- [x] Modify：先预测d2→d1，再运行核对数据不变量、shape与物理误差，并查看aliasing_probe。
- [x] Explain：用自己的话回答五问，画E→F→D与future target两条分支。

Engineering Complete + Learning Mastered（2026-10-20 用户明确确认实验和预测完成，并准确回答五项Explain）。
解释覆盖E/D/F的shape和单位、PCA拟合后冻结、630观测/600转移与train-only统计、
重建/latent/物理误差和特权reference、d1 aliasing及动力学充分性、AR未来观测隔离与toy证据边界。
接口补充：D输出六路观测；已知sensor伪逆是独立的物理评价步骤。
Current Task=S3.4；Next Task=S3.5仅计划。
STOP，不自动开始下一任务。
