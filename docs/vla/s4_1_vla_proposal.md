# S4.1 — VLA Proposal

## Problem

把冻结S1.5 tiny VLA接到显式候选文件接口，检查动作语义，为后续验证留边界。
本任务只生成proposal；S4.2才预测未来。不用S1.6 oracle bridge替代失败策略。

## Intuition

策略交出一份三步计划，并不意味着该计划已获准执行。先问“它是什么命令、针对哪个系统、从何时开始”。
同样是(3,2)，world位移、camera位移与绝对目标位置不能互换。
确定性策略只有一个输出；重复调用不会自动获得有意义的多模态候选。

## Input / Output Contract

[脚本](../../scripts/s4_1_vla_proposal.py)，文件schema `vla_proposal/1.0`。

| 项 | shape / dtype | 单位、frame、时间与来源 |
|---|---|---|
| RGB | (32,32,3) uint8 | 当前静态合成正交视图，无agent marker；无未来帧 |
| state | (2,) float32 | 当次观测world XY位置[m]；本实验tick0=[0,−.12] |
| language | string | 原literal `move to red/blue`；其他文本保留拒绝 |
| image patches | (1,16,5) float32 | 4×4平均RGB+patch坐标，无量纲 |
| words / state token输入 | (1,3) int64 / (1,2) float32 | 原词表；state除以.3m，固定原训练尺度，无重新fit |
| proposal bank | (K,3,2) float32 | world XY commanded displacement[m]；每分量±.06m；dt=.1s |
| batch文件action | (16,K,3,2) float32 | 8旧validation layouts×2语言；每例从tick0同时提出候选 |
| batch image/state/text/tick | (16,32,32,3)/(16,2)/(16,)/(16,) | uint8/float32/Unicode/int64；NPZ禁用pickle |

Embodiment为`ideal_world_xy_patch_grid_v1`。slot h对应t+h→t+h+1，整段只条件于t观测；
未来观测、goal坐标、成功标签均不进入proposal函数。H=3，时间跨度.3s；K是候选数，不是时间。
proposal没有padding，每槽是真实候选命令；这些不是已执行trajectory，不能冒充T/T+1 episode。
S3.6墙环境的action即使shape/frame/unit相同，也不能直接认定embodiment一致；本轮不连接它。

## Core Concepts

- candidate0是冻结learned输出（原head×.06m并clip）；之后是显式手工偏置候选。
- bias bank固定为[0,0]、[.01,0]、[−.01,0]、[0,.01]、[0,−.01]m，沿三个slot广播。
- 这不是learned stochastic sampling、重新训练或多模态能力；clip后可能重复，report记录unique数。
- K采用固定前缀顺序；减少K不改变candidate0、权重、输入或闭环基线。
- Contract拒绝错误version/embodiment/语义/frame/unit/dt及shape/dtype/NaN/越限。
  检查声明无法证明数据生产者如实标注，更不能保证物理安全。

## Mathematics

B₀=clip(.06·fθ(RGBₜ,ℓ,pₜ/.3),−.06,.06)，shape(3,2)，单位m。
Bₖ[h,j]=clip(B₀[h,j]+bₖ[j],−.06,.06)。bₖ为以上固定world XY偏置[m]。
θ、tokenizer、固定尺度均冻结；没有loss反传或新normalizer。

单步每分量限制.06m，并非L2长度限制：对角线最大长度√2×.06m。
没有评分函数或argmin；候选数增加不能据此推断success改善。

## Architecture / Data Flow

五份S1.5 source artifacts→hash/readonly loader→当前RGB/语言/state→patch+word+proprio Transformer
→H3物理chunk→固定bias bank+clip→contract检查→NPZ/JSON。独立evaluator只诊断candidate0闭环。

## Math / Architecture → Code

`propose`实现B₀与bias广播；`validate_proposals`检查显式文件语义及数组。
复用S1.7 loader、offline_metrics与run_case，不修改原策略。source五文件前后SHA256一致，weights readonly。
proposal latency包括patch/tokenizer/推理、bank和validation；不含render/IO，5次warmup排除。
闭环诊断单独测量，不混进proposal latency。

## Minimal Experiment

默认source `tmp/s4_1_source_s1_5`；若已有自己的S1.5目录，两条都加同一`--checkpoint`。
缺失source会报错，不隐式训练。此次本地历史checkpoint缺失，已显式使用原脚本重建：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s1_5_tiny_transformer_vla.py --output tmp/s4_1_source_s1_5
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_1_vla_proposal.py --output tmp/my_s4_1
# 先预测K5→1，再运行新目录：
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_1_vla_proposal.py --candidates 1 --output tmp/my_s4_1_k1
```

重建使用原默认seed7/2400steps，小型CPU NumPy合成训练；无下载，无新依赖。
不声称重建权重文件与历史Python3.14运行逐bit一致。本次实际来源hash见report。
看contract.json、proposals.npz、report.json；所有生成物留ignored tmp。

## Expected Result

K5有16×5×3×2动作，K1有16×1×3×2。candidate0逐值不变；不支持文本仍拒绝。
动作合法并不改变已知策略失败，proposal没有选择/执行其他候选。

## Actual Result

2026-10-20，repo .venv Python3.12.3 / NumPy2.5.4 CPU，模块在本repo .venv。
重建S1.5成功，之后两次proposal均无训练。16/18 proposal attempts生成候选，2/18输入拒绝；
拒绝无伪造chunk，不能称为2个“坏动作被风险过滤”。
K5/K1 action shape分别(16,5,3,2)/(16,1,3,2)，candidate0与输入逐值相同。
独立candidate0 E1闭环两次均0/16成功，预算8步；失败stage为final_goal_error_after_budget。
原expert-state masked offline MSE保留在report，与visited-state误差分开。
K5 proposal median/P95 .134/.231ms，K1 .139/.190ms，均16次，非稳定速度排名或整机实时保证。
3项边界测试通过；跨K输入/前缀/source/offline及闭环commands不变量通过。

## Result / Explanation

建立了可审计的候选接口，未改善控制能力。手工候选不保证多样性、可达性或比candidate0更好。
闭环结果来自candidate0在ideal point环境的执行诊断，不是整个候选bank被选择后的success。

## Failure Analysis / Limitations

| 失败或限制 | 处理与解释 |
|---|---|
| unknown文本 | 保留拒绝记录，不接oracle补语义 |
| normalized/absolute/camera动作误接 | 严格contract声明检查，生产者真实性仍需审计 |
| bias饱和/重复 | clip后记录unique数量；K不是不同策略数 |
| 合法但错误的chunk | candidate0闭环仍失败，未被接口检查修复 |
| reused validation | 8布局不是新独立test，无新泛化结论 |
| model/plant错配 | 本轮不接墙world model，不偷偷改变embodiment |
| latency波动 | 本地计算边界明确，无sensor/servo等待或实时保证 |

## Relation to Robotics

这是policy与后续verifier间的文件边界。真实接入需核对controller、frame、dt、anchor与embodiment；
不导入或修改sibling repo。没有真实机器人、接触动力学或风险校准证据。

## Interview Capsule / Notes

30秒：冻结tiny VLA，生成原H3位移及四个手工偏置候选，保留来源并严格检查m/world XY/dt/embodiment。
合法接口不等于成功，candidate0仍0/16；未接world model或选择器。

2分钟：解释H/K/E区别、bias和clip、readonly/hash、future/goal信息隔离、unknown拒绝分母，
以及为什么相同shape的S3墙模型不能直接插上、为什么旧validation和本地latency不能证明部署能力。

## Likely Follow-up Questions

1. 输入模态、dtype、shape、单位和时刻是什么？slot h是否看到了未来RGB？
2. candidate0与其余候选各来自哪里？重复推理为什么不等于learned多模态采样？
3. K5→1哪些量必须不变？H、K、E和dt分别控制什么？
4. 为什么contract通过仍可能失败？world XY相同为何不能直接接S3墙模型？
5. 16/18提案生成与0/16闭环成功分别是什么分母？两种拒绝、offline loss及latency不能证明什么？

## Must Remember

proposal≠执行批准；合法≠成功；手工bias≠learned采样；shape相同≠embodiment一致；K≠H≠E。

## My Verification

- [ ] Run：新目录运行，检查contract、action数组与拒绝记录、candidate0闭环失败。
- [ ] Modify：先预测K5→1对candidate0、输入、source hash、offline、闭环commands及shape的影响，再运行核对。
- [ ] Explain：用自己的话回答上述五问。

Engineering Complete；Learning未确认。Current Task=S4.1；Next Task=S4.2仅计划，STOP。
