# S4.2 — World-model Verification

## Problem

读取冻结S4.1候选文件，预测每条chunk的未来，与独立plant实际执行的未来逐步对照。
这里的verification是预测误差核验；不是风险批准、候选选择或形式化安全证明。
本任务不进入S4.3评分，不改变VLA，S4.1 candidate0闭环0/16的证据不被覆盖。

## Intuition

从同一起点，把同一份三步命令分别交给模型和plant：模型只能拿自己的预测继续往前走；
plant每步产生真实状态。每个候选都重新reset，不能把候选1接到候选0终点。
另设teacher forcing诊断：每步给模型真实的执行前状态。这能测局部误差，却不是起点时可用的完整未来预测。

## Input / Output Contract

[script](../../scripts/s4_2_world_model_verification.py)，[tests](../../tests/test_s4_2_world_model_verification.py)。
输入是S4.1 `contract.json`、`proposals.npz`和`report.json`；默认`tmp/s4_1_verified`，
可以用`--proposals tmp/my_s4_1`选自己的已完成产物。缺失即报错，不偷偷重建策略/下载。

| 数据 | shape / dtype | 单位、frame、时间与流向 |
|---|---|---|
| frozen proposals | (16,K,3,2) float32 | commanded world XY displacement[m]，每分量±.06m；H3/dt.1s |
| current position | (16,2) float32 | tick0实测world XY[m]，广播到K候选；无未来状态 |
| current RGB/text/tick | (16,32,32,3) uint8 / (16,) Unicode / (16,) int64 | 保留原proposal输入并校验；动力学不读取RGB或语言 |
| dynamics input | (...,4) float64 | [x,y,a_x,a_y]原物理m，减train mean除train std后无量纲 |
| design / coefficients | (...,5)/(5,2) float64 | 4路归一化数值+常数1；10个拟合系数把输入映射为delta[m] |
| AR trajectory | (16,K,4,2) float64 | 初态+递归预测3个未来world XY状态[m]；不读真实未来 |
| independent plant truth | (16,K,4,2) float32 | 每候选独立reset，执行3条相同command；包含初态 |
| TF prediction | (16,K,3,2) float64 | 从真实state[h]预测state[h+1]；仅事后诊断，无初态槽 |
| clock | (4,) float64 | [0,.1,.2,.3]s，模拟时间，与wall latency无关 |
| training raw states/actions | (40,21,2)/(40,20,2) float32 | 40独立合成episode，每条20transitions，原物理量保留 |

输出`contract.json`采用`candidate_future_verification/1.0`，内含
`ideal_xy_affine_dynamics/1.0` model contract与原proposal contract。
`verification.npz`保存动作、初态、AR、zero-action消融、两条件真值/TF与时钟；不使用object/pickle。
`training_episodes.npz`保存完整raw训练/验证episodes及split IDs。
模型checkpoint `model.npz`保存系数及仅用train拟合的mean/std；report含SHA256。

## Core Concepts

- 原S3.1有位置相关gain，S3.6有gain/墙约束；均与S4.1 ideal plant不同，不直接接入。
- 此处新拟合**匹配ideal point的线性残差模型**，不声称复用S3 checkpoint或训练大型world model。
- 名义plant为p_next=p+a；模型类恰好包含真规律，因此近零误差是简单可识别系统的结果。
- 训练用独立随机commands，无VLA候选、RGB、语言、goal或success；VLA保持冻结。
- 拟合后save/reload、参数readonly，所有候选预测完成后才生成其执行真值。
- gain shift是显式evaluation-only机制变化：执行p_next=p+g·a，模型不知道g，仍假设名义gain1。
  不是悄悄修改action单位或把realized delta重新当command。
- 预测只输出位置；没有goal distance、reward/success/risk head、阈值、argmin或fallback。

## Mathematics

输入x=[p_x,p_y,a_x,a_y]，train-only μ,σ∈R⁴。
φ(x)=[(x−μ)/σ,1]∈R⁵，目标y=p_next−p∈R²[m]。

W*=argmin_W Σ_train ||φ(x)W−y||²，W∈R^(5×2)。
用`np.linalg.lstsq`求解，训练design rank=5；没有梯度optimizer或epoch超参数。
目标保持物理m，不fit target normalizer。30train episodes共600转移；10validation episodes共200转移。
seed421生成数据、seed422整episode划分；先split再展平，validation与proposal不参与fit。

递归预测：p̂₀=p₀；p̂_(h+1)=p̂_h+φ([p̂_h,a_h])W。
Teacher forcing：p̃_(h+1)=p_h^actual+φ([p_h^actual,a_h])W。
独立plant：p_(h+1)^actual=p_h^actual+g·a_h；gain是plant参数，**不进入predict_next/imagine**。

忽略拟合/浮点误差时，预测仍按gain1：

- AR误差 e_(h+1)=(1−g)Σ_(i=0…h)a_i。
- TF误差 e_TF_(h+1)=(1−g)a_h。

AR误差取决于动作累加，动作相消时不必逐步单调增长。本实验实际曲线增长，不推广为一般定理。
g从.65→.4后，gain偏差.35→.6，误差近似乘12/7；模型和AR预测不变，只改真值和TF轨迹/误差。

RMSE=√(Σ_(case,k,h,j)e² / (16·K·3·2))，单位m；K5分母480，K1分母96。
始终剔除共享初态，避免额外零误差稀释。每horizon RMSE分母16·K·2。
末步另报L2误差P95[m]；不是每坐标RMSE或整机latency。

## Architecture / Data Flow

S4.1文件→严格contract与payload校验→只读候选/初态→冻结model AR3→预测产物。
独立训练支路：random reset episodes→episode split→train statistics/least squares→checkpoint重载冻结。
独立评价支路：相同初态/commands→名义或gain-shift plant→真值→与AR比较；
真值另供TF诊断，不反馈到AR。原2次unknown语言拒绝记录原样保留，不造未来。

## Math / Architecture → Code

`load_proposals`检查原schema、embodiment/单位/frame/dt、16案例、K、dtype、tick0及拒绝分母。
`training_episodes`、`fit_model`对应数据/split/线性最小二乘；`validate_model`拒绝错plant声明或坏统计。
`predict_next`只接current/command/model；`imagine`递归，不接gain或truth。
`plant_rollout`独立更新float32真值；`errors`汇总剔除初态后的物理误差。
`main`保存raw/model/source hashes、完整未来和curves.csv；源文件前后hash及输出round-trip均检查。

## Minimal Experiment

CPU repo .venv，只有已有NumPy，无新依赖/安装/数据或模型下载。
每次显式生成小型合成训练集并拟合一次，随后冻结；第二次也是相同seed/data/split重新拟合，非读取上次模型。
两个run必须使用相同proposal来源，先读公式并预测gain变化后再执行第二条：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_2_world_model_verification.py --output tmp/my_s4_2
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_2_world_model_verification.py --plant-gain .4 --output tmp/my_s4_2_g04
```

若使用自己的S4.1目录，两条均加`--proposals tmp/my_s4_1`。
查看contract.json、report.json、verification.npz、curves.csv、model.npz和training_episodes.npz。
训练集起点±.4m、命令±.06m；没有真实机器人数据、图像生成或动作重新规划。

## Expected Result

名义ideal系统的learned prediction接近float32数值舍入误差；zero-action消融明显变差。
未知gain导致AR预测与真实执行偏离，TF局部误差与AR累计误差不同。
gain .65→.4不改变source/model/statistics/原始训练数据/AR预测/名义真值，改变shifted真值、TF与误差。

## Actual Result

2026-10-20，repo .venv Python3.12.3 / NumPy2.5.4 CPU，NumPy实际路径在本venv。
默认K5：16个supported inputs×5候选=80条counterfactual rollout/条件，每条3命令、4状态；
两个条件共160条，不是160个独立场景。原18次proposal输入中的2拒绝只传递记录。
30train/10validation、600/200转移；held-out single-step RMSE=6.823×10⁻⁹m。
K5名义AR全未来RMSE=2.409×10⁻⁹m；zero-action消融RMSE=23.984mm。

| 条件 | AR step1 RMSE mm | AR step2 RMSE mm | AR step3 RMSE mm | AR全未来RMSE mm | TF全转移RMSE mm |
|---|---:|---:|---:|---:|---:|
| gain1 nominal | .000001915 | .000002574 | .000002668 | .000002409 | .000001624 |
| gain.65 | 2.498 | 6.443 | 12.792 | 8.394 | 4.813 |
| gain.4 | 4.283 | 11.046 | 21.930 | 14.391 | 8.251 |

K5默认gain.65末状态L2 P95=27.332mm。30次batch80 AR3 latency median/P95=.0258/.0343ms，
5warmup排除，仅normalize/matmul/update，不含proposal/plant/评分/IO/真实通信；无实时性能结论。
额外K1回归每条件16条轨迹：保留每case candidate0全部轨迹，因候选集合/分母改变，汇总误差可不同。
5项新增测试通过；独立解析核对AR累加/TF局部误差、每候选reset、gain/K不变量、train-only统计、
模型与source文件不变、NPZ round-trip通过。完整验证与命令见handoff。

## Result / Explanation

在匹配系统中，模型能准确预测**错误策略提出的动作会把点带到哪里**；不等于策略朝正确目标运动。
机制变化能在不改模型/动作/初态的情况下造成明显预测偏差。
本节没有候选接受或排序，也没有部署新controller；不报告虚构闭环success，report该字段为null。
S4.1的candidate0闭环0/16仍是原有策略能力证据，不将prediction accuracy当成控制进步。

## Failure Analysis / Limitations

| 情况 | 影响与证据边界 |
|---|---|
| 不同plant声明 | contract严格拒绝；不能因shape相同接墙模型 |
| 未知机制变化 | gain shift违背名义模型假设，误差明显；没有在线检测或自动校准 |
| TF真值泄漏入AR | 测试隔离；TF是特权诊断而非可部署起点预测 |
| 候选串接执行 | 每候选独立reset，保存初态与T/T+1时序 |
| 验证泄漏 | 改validation states/actions不会改变fit或train统计；proposal从未进入训练 |
| 超简单模型类 | 真plant是线性无噪声可完全观察系统，近零误差不证明非线性/接触/视觉能力 |
| 重复场景/偏置候选 | 旧8layouts×2文本，候选相关，无独立VLA test或统计显著性声明 |
| 任务成功/风险 | 未定义本节评分器或校准阈值，无bad-action拒绝/误拒率 |
| 多步长度 | 仅3步.3s；无长时稳定/开放世界泛化保证 |

## Relation to Robotics

连接proposal文件和状态预测的可审计边界；真实系统还需估计器误差、时间延迟、控制器与接触模型。
物理command、realized delta与外界机制不能混为一谈；sibling repo未改动或导入。

## Interview Capsule / Notes

30秒：我为冻结VLA候选匹配一个10参数ideal-point残差模型，train-only拟合后冻结。
每候选独立reset，对比AR预测与真实执行；名义近数值误差，gain.65使末步RMSE12.79mm。

2分钟：说明为什么不用S3墙模型、TF为什么看真值却不能代替AR、未知gain怎样在模型不变时
产生(1−g)Σa误差，以及预测正确为何不能说明candidate好或闭环成功。

## Likely Follow-up Questions

1. policy读取哪些模态，dynamics实际读取哪些？动作/AR/truth/TF的shape、dtype、时序与单位各是什么？
2. 为什么要新拟合匹配ideal plant的模型？10参数是什么，训练/验证如何分组，normalizer如何避免泄漏？
3. AR和TF每步输入哪里来？为什么实际未来只能用于评价/TF？每个candidate为什么必须reset？
4. gain.65→.4哪些量不变、哪些会变？推导AR与TF误差，为什么AR误差不保证随步数单调增长？
5. 18次输入、80条候选rollout、480个误差scalars分别是什么分母？近零prediction误差与本地latency不能证明什么？

## Must Remember

匹配plant；train-only fit；freeze后预测；AR不读真值；候选独立reset；prediction准确≠任务成功。

## My Verification

- [x] Run：新目录运行默认命令，检查四状态三动作、gain.65的AR/TF误差及原拒绝记录。
- [x] Modify：先预测gain.65→.4不变量、误差近似12/7倍及TF变化，再运行新目录核对。
- [x] Explain：独立回答上述五问，手算80×3×2=480并区分counterfactual rollout与闭环控制。

Engineering Complete + Learning Mastered（2026-10-20 用户确认）。Current Task=S4.2；Next Task=S4.3 — Risk / Success Scoring仅计划，STOP。

2026-10-20 首次学习核对：用户明确确认实验和预测均完成，正确解释AR/TF因果边界、候选独立reset、gain误差推导、
12/7倍率、分母480/160和能力证据限制。当时Run/Modify已勾选；Explain暂留未勾选，需补充：
truth固定float32，AR固定float64，两者均(16,5,4,2)；TF固定(16,5,3,2)float64，不含初态。
单次VLA输出(3,2)，候选函数单例输出(5,3,2)，16例汇总才为(16,5,3,2)。
本模型为phi=[(x−train_mean)/train_std,1]，delta=phi@coefficients，coefficients为(5,2)float64；
30train/10validation独立episodes、600/200transitions，最小二乘拟合10系数，目标delta保留m。
用户给出的2×4矩阵+2维bias是等价仿射写法，但本文件采用上述归一化行向量参数化。

2026-10-20 用户随后准确补充：单例learned chunk→手工bank→16例汇总shape；
AR/truth均4states但分别float64/float32，TF为3个float64单步预测且不含初态；
phi=[(x−mu_train)/sigma_train,1]、delta=phi C、C为(5,2)共10系数；
30train600/10validation200、完整episode划分、仅train拟合normalizer及系数。
结合此前正确的第3–5项Explain，Run/Modify/Explain全部勾选，Learning Mastered。
不增加候选选择、闭环改善或真实机器人能力声明；S4.3仅计划。
