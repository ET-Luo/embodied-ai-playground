# S3.8 — Video / Generative World Model Concepts

## Problem / Why

S3.1–S3.7讨论state/latent预测与反馈控制。视频world model还要生成高维观测、
处理部分可观测下的多种未来，并保持动作响应与跨帧一致性。
本包用固定版本官方资料学习三种架构，再用CPU解析反例区分像素误差、轨迹正确性与可控性。
不训练/运行大型视频模型，不下载dataset或权重，不声称复现Dreamer、GAIA或Genie。
预计1.5小时：官方架构阅读25分钟，公式/代码25分钟，Run/Modify20分钟，Explain20分钟。

## Intuition

未来不是永远只有一条：同一初始画面和动作，在隐藏状态未知时可能向上或向下漂移。
对两种未来取平均，可能得到不存在的中间运动，或者一个物体变成两个淡影。
每帧单独看都可能合理，拼成一段却可能跳跃；动作若被忽略，画面再好看也不能用于动作选择。

## Official Architecture Study / Version Boundary

2026-10-20查阅，精确链接与访问边界见 [source manifest](s3_8_sources.json)。
以下为固定历史变体，不是最新模型排名；官方效果与本地反例是不同证据层。

| 固定变体 | 架构主线 | 学习重点 / 边界 |
|---|---|---|
| DreamerV3，arXiv v2 (2024-04-17) | observation→RSSM表示；动作条件latent imagination；actor/critic在想象轨迹上学习 | latent world model可用于控制，不必每步先生成展示视频；不是diffusion video generator |
| GAIA-1，arXiv v1 (2023-09-29) | video/text/action编码为tokens→AR预测image tokens→时序video diffusion decoder | token dynamics与视频渲染分工；不是GAIA-2，也不是robot delta-action统一接口 |
| Genie 2，官方介绍 (2024-12-04) | prompt image/过去latent frames + keyboard/mouse action→causal Transformer latent diffusion→frames | 时间AR与每步扩散可并存；不借用Genie 1 latent action或Genie 3特性 |

依据：[DreamerV3 paper](https://arxiv.org/html/2301.04104v2)、
[GAIA-1 Figure 2 / Model / Video Decoder](https://arxiv.org/html/2309.17080v1)、
[Genie 2 Diffusion world model](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)。
Genie博客是作者展示与架构说明，不是我们审计过的物理准确率benchmark；没有在本机执行这些上游系统。

**Dreamer代码阅读**：官方repo HEAD经git ls-remote固定为
`e01491fad6434b2245a3b8ca201dd7faedcc458c`。
[`RSSM.observe/_observe`与`imagine`](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/rssm.py)
分别使用观测token或动作驱动的prior；
[`Agent.loss`](https://github.com/danijar/dreamerv3/blob/e01491fad6434b2245a3b8ca201dd7faedcc458c/dreamerv3/agent.py)
可追踪重建/reward/continuation与imagination分支。
这是所选函数静态阅读，不能假定当前revision与2024论文逐行相同；未安装JAX或运行训练。

## Input / Output Contract

本地新profile `s3_8_branching_video/1.0`，与S3.5墙toy不同：无墙、无contact、无任务goal。
它是**已知解析sampler+renderer**，没有learned encoder、dynamics、denoiser或视频decoder。

| 项 | shape / dtype | 单位、frame、时间 |
|---|---|---|
| initial position | (20,2) float32 | world XY[m]，每轴uniform[-.015,.015]，seed380 |
| candidate actions | (20,2,8,2) float32 | 左/右两个条件；每步a_x=−/+ .015m，a_y=0，dt=.1s |
| true branches | (20,2 actions,2 modes,9,2) float64 | 相同初态的两种固定隐藏mode b=−1/+1，等概率；world XY[m] |
| sampled futures | (20,2,K,9,2) float64 | 默认K16；mode采样seed38000+reset*1000+sample |
| truth images | (20,2,2,9,32,32,3) uint8 | RGB 0…255，每帧一个3×3橙色square |
| pixel mean | (20,2,9,32,32,3) float64 | 两条真分支的精确条件像素均值，特权解析参考 |
| timestamp | (9,) float64 | 相对秒，8转移/9观测，终点无动作 |

world范围±.25m；+x映射图像列向右，+y映射行向上，nearest pixel舍入。
背景RGB=[12,16,24]、物体=[245,100,60]；renderer遇越界明确拒绝，不偷偷clip。
所有context在同reset内跨action/mode完全相同；action_t影响frame_t→frame_t+1。
初始XY直接由generator提供，是**特权数值初态**，没有从RGB学习/推断位置的encoder。
采样只用初态、给定动作和随机seed，不看真实未来分支标签；真分支用于评价和均值参考。

每条episode8步后time truncation，无success/failure任务标签。无force、velocity、语言或机器人控制器。
没有训练或拟合统计，因此不伪造train/test、held-out loss或模型泛化结论；全部20个reset都是解析探针。

## Core Concepts

- Generative future：表达一个条件分布，不只是给一个未来均值。
- Action conditioning：同context改变action后，预测分布应有相应变化；仅拼接action不保证模型真正使用它。
- Conditional plausibility与正确物理joint不同：单帧合理，不代表整段状态/物体身份持续一致。
- 画面MSE/感知质量、动作响应、temporal consistency、物理残差、任务成功应分别评价。
- 隐藏mode造成从初始观测看未来不确定；完整状态含b时这里仍是确定性系统。
  不能把它等同ensemble epistemic disagreement，也不能声称已估计真实不可约噪声。
- 多样性大不等于准确，best-of-K改善也不等于单次可部署预测变好。
- 学到条件相关性不自动识别真实动作干预因果；本toy动作效应只因generator明确定义而可验证。

## Mathematics

### 生成式时序与latent

通用动作条件未来分解为：

\[
p(o_{t+1:t+H}\mid o_{\le t},a_{t:t+H-1})=
\prod_{h=1}^{H}p(o_{t+h}\mid o_{\le t},o_{t+1:t+h-1},a_{t:t+h-1}).
\]

这里h是物理未来帧；AR训练可以用真前缀做teacher forcing，部署只能用已生成前缀。
扩散内部的denoising index τ是另一条时间轴，不等于h或机器人dt。
通用latent pipeline是`context→encoder→latent dynamics conditioned on action/noise→decoder→future video`。
有些系统在latent里完成policy/value学习，不需要把所有想象轨迹先渲染成RGB。

与Dreamer式观测/想象区别相符的简化记法：
\(h_t=f(h_{t-1},z_{t-1},a_{t-1})\)，观测时用
\(z_t\sim q(z_t|h_t,o_t)\)，未来想象用\(z_t\sim p(z_t|h_t)\)。
未来真图像只可用于监督或观测更新，不可混入被声称为open-loop imagination的prior分支。

### 教学tensor trace（不是上述模型的实参）

假设B2、context2帧、future8帧、RGB32×32；8维latent grid用stride8压成4×4：
`context (2,2,32,32,3) → z_context (2,2,4,4,8)`，
`actions (2,8,2) → action embeddings → predicted latent (2,8,4,4,8) → frames (2,8,32,32,3)`。
噪声tensor可与future latent同shape；latent无量纲，原action必须另外声明单位/frame/dt。
同shape不代表相同动作语义；temporal upsampling也不意味着新的独立控制指令。
时空token数随帧数和空间grid增长，attention可有二次成本；本任务不据此假装测过大模型latency。

### 本地解析规则

\[
p_h=p_0+\sum_{j=0}^{h-1}a_j+[0,b\,h\delta],\quad b\in\{-1,+1\},\quad\delta=.0075m.
\]

b在整个episode固定；p0未知b但相同画面可产生两个未来。两个动作条件的8步x终点差=.24m。
latent sampler这里就是选b并执行已知公式；renderer也是手写，不是学习到的世界知识。

四类对照：

1. **persistent**：每个sample只采一次b，整段保持，合法trajectory。
2. **frame_resampled**：每个绝对future h独立采b_h，位置仍选该帧±hδ。
   每个时刻的边缘分布与合法prior相同，但跨帧joint被破坏；它不是另一种合法iid过程噪声环境。
3. **mean_state**：b取0，条件期望position再render；MSE可小，却不是任何合法分支。
4. **action_ignored**：用合法固定b但把输入action丢弃，无法产生正确左右动作响应。

另有exact **mean pixels**：两真分支图像平均，均方像素误差最优参考却可产生ghost objects；
与`render(mean state)`不同，因为renderer非线性。copy-initial-frame是稀疏目标背景捷径基线。

### 物理与分布评价

合法转移满足 \(\Delta x-a_x=0\)、\(|\Delta y-a_y|=\delta\)。
以1e−8m容差统计temporal violation，不从画面漂亮程度推断物理正确。
raw state path MSE对8个future slots×2坐标平均，单位m²，t0不稀释分母。
pixel MSE将RGB值除255后对未来8帧/所有像素通道平均，无量纲。

all-sample path MSE对20 reset×2 action×2真branch×K sample平均；
best-of-K为**给定完整真实未来后**选择整段误差最小sample，再对80条真轨迹平均，属于特权诊断。
exact support coverage是80条真轨迹中，样本bank包含整条匹配轨迹的比例，不是校准概率或控制success。

对称两分支的条件均值：
\(MSE_{mean}=\delta^2\operatorname{mean}(1^2,\ldots,8^2)/2=.0007171875m^2\)。
固定采样分支与等概率真分支比较，all-sample MSE为其两倍；合法sample不必赢过非法均值的paired MSE。
本toy的两个已知support都已写出；coverage100%不证明学习或真实开放世界覆盖。

## Architecture / Data Flow → Code

[脚本](../../scripts/s3_8_generative_world_concepts.py)：`build`产生配对context/actions/已知支持与随机sampler；
`positions`执行解析latent rollout，`render`生成RGB，`temporal_violation`检查规则，
`sample_summary`分开all-sample与best-of-K，`previews`固定展示reset0/right/sample0，不挑最好看sample。
`component_count`只在这个已知背景/物体toy中统计连通块，不是通用vision evaluator。

`trajectories.npz`保留float32初态/命令、float64轨迹、sample signs、truth RGB、条件像素均值与时钟；
`report.json`保存完整指标，`contract.json`声明语义，`comparison.png/gif`供观察。
无需外部模型或数据；NPZ读取使用`allow_pickle=False`。
[测试](../../tests/test_s3_8_generative_world_concepts.py)核验相同context的动作反事实、时间joint、
K样本前缀/覆盖与单次误差区别、像素均值的ghost objects。

## Minimal Experiment

repo .venv Python3.12.3 / NumPy2.5.4 / Pillow12.3.0，CPU，仅已有依赖，无安装/数据或weights下载。
官方论文和网页用于阅读；没有下载其视频样本、仓库或上游运行环境。
输出目录必须不存在：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_8_generative_world_concepts.py --output tmp/my_s3_8
# 先预测K16→1对raw真值、首个sample、覆盖、best-of-K和单次误差的不同影响：
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s3_8_generative_world_concepts.py --samples 1 --output tmp/my_s3_8_k1
```

打开PNG（每行从frame0至8）和GIF：两条真branch、persistent sample0、frame-resampled sample0、
mean state render、mean pixels。然后对照report，不能只凭视频观感下结论。

## Expected Result

persistent规则残差应0，action-ignored/mean-state违例率100%；frame-resampled边缘相同但大部分整段不合法。
K减少保持真实初态/动作/分支/图像、均值参考与sample0不变；coverage不能因移除样本而提高，best-of-K不能改善。
更多样本没有改变单次sample质量；oracle挑样本会使best-of-K看起来更好，但部署前无法知道哪条匹配实际未来。

## Actual Result

2026-10-20，20 reset/40 action contexts/80真轨迹，K16：

| 解析预测器 | all-sample path MSE [m²] | best-of-K [m²] | exact trajectory coverage | transition violation |
|---|---:|---:|---:|---:|
| persistent | .001434375 | 0 | 100% | 0% |
| frame-resampled | .001434375 | .000354727 | 7.5% | 44.5703% |
| action-ignored | .004303125 | .002868750 | 0% | 100% |

frame-resampled有99.0625%的生成整段含至少一次违例；与persistent具有相同all-sample MSE却不同物理joint。
mean-state MSE=.0007171875m²，比合法sample更小，但转移违例100%。
mean-pixels/mean-state-render/copy-context的pixel MSE分别.001211227/.003733637/.005030121；
末帧真值1个物体，pixel mean出现2个淡影。物体只有9/1024像素，占比小，背景主导像素平均。
左右动作终点相距约.24m；action-ignored不能实现该响应。

K1：persistent coverage50%、best-of-K=.001434375；K16 coverage100%、best-of-K0，
但all-sample MSE相同且单个sample轨迹逐值不变。frame-resampled coverage为0%，违例48.125%。
K1的20次mode抽样正分支比例30%，K16为50.625%，有限采样比例不等于理论prior改变。
所有物理/样本前缀与像素参考均可重现；PNG已检查，GIF可解码，不是上游模型生成结果。
31项Stage3协议测试通过（此前27项+本轮4项）。无learned model，故不报告虚构训练loss、模型latency或机器人success。

## Explanation / Result

这些解析反例分别说明：动作条件必须验证，边缘像真不代表时序joint对，均值/像素损失不等于有效未来，
多sample的oracle指标不等于实际可控性。它们不证明任何官方模型有相同失败率，也不复现其机制。
本包没有真实world model训练、OOD泛化或闭环控制实验；大模型的物理可靠性需要独立匹配数据/动作/任务评价。

## Failure Analysis / Failure Cases

| 误读 | 本包边界 |
|---|---|
| generative等于一定用diffusion | AR、stochastic latent、diffusion是不同维度，可组合；固定版本分别说明 |
| video流畅等于正确动力学 | 分开检查action response、物体数、joint与规则残差 |
| 每帧都合法所以整段合法 | frame-resampled逐帧支持正确，跨帧却跳跃 |
| mean像素最优就是好simulation | 两淡影不是真实单物体状态；mean-state也不属于合法support |
| 单个sample未匹配真值就是模型差 | 多模态未来需看分布/条件和joint，配对MSE不能独自裁决 |
| best-of-K0就是部署完美 | 用未来真值挑sample，明确是特权指标 |
| 随机branch等于ensemble uncertainty | 这里只是初始隐藏mode，完整状态下确定；无学习不确定性估计 |
| toy renderer冒充learned decoder | 已知公式/renderer，无视觉encoder、denoiser或神经采样 |
| 官方演示冒充本地证据 | 固定manifest，论文/源码/博客/本地反例分开，不宣称复现 |
| Stage3包完成就进Stage4 | 仍需用户学习确认，S4.1仅计划，未自动实施 |

## Relation to Robotics / Evaluation Checklist

真正用于机器人时，至少区分动作语义/时间对齐、空间/物体持久性、动作反事实、长时误差、
分布覆盖/校准、OOD与闭环任务成功、端到端latency；video metric不替代这些证据。
S3.7状态反馈有帮助，也不能纠正所有视频幻觉或未知contact机制；本轮不把视频样本接入控制。

## Interview Capsule / Interview Notes

30秒：我区分Dreamer式latent imagination、GAIA-1 token dynamics/video decoder和Genie2动作条件latent diffusion。
本地只做已知解析采样：逐帧重抽分支与合法持久分支有相同平均MSE，却有约44.6%转移违例；
pixel mean的误差更低但产生两个ghost物体。best-of-K用未来真值挑样本，不是可部署预测或机器人成功。

2分钟：补充context/action/noise的条件分布、posterior与prior、物理帧时间与denoising τ、
教学tensor trace、隐藏mode与随机过程差别，最后说明paper claims/静态源码/本地toy三种证据。

## Likely Follow-up Questions

1. DreamerV3、GAIA-1、Genie2的world-model/decoder/action路径有何区别？为什么generative不等于diffusion，latent imagination也不必先render视频？
2. 本地初态/动作/hidden branch/轨迹/图像分别是什么shape、单位与时序？b为何不等于每帧iid噪声或ensemble disagreement？
3. 为什么mean-state/pixel mean的MSE较小，却不属于合法单物体未来？frame-resampled为何边缘正确而joint错误？
4. K16→1哪些数据/样本保持不变？coverage和best-of-K如何变化，为什么不能把best-of-K当真实单次预测或部署success？
5. prior imagination如何避免未来图像泄漏？从这个解析反例或官方视频，不能推出哪些物理/机器人控制结论，需要哪些独立评价？

## Must Remember

generative≠diffusion；pixels≠physics；frame marginals≠trajectory joint；best-of-K≠deployment；纸面架构/官方展示/本地解析证据不可互换。

## My Verification / Learner Verification

- [x] Run：阅读固定版本架构表、运行K16，查看PNG/GIF与物理/像素指标。
- [x] Modify：先预测K16→1的不变量及coverage/best-of-K变化，再运行核对。
- [x] Explain：用自己的话回答五问，区分真正模型学习与本地解析示例。

Engineering Complete + Learning Mastered（2026-10-20 用户明确确认实验和预测完成，并准确回答五项Explain）。
解释覆盖固定变体架构、generative/AR/diffusion区别、hidden mode的shape/单位/时间、
均值与边缘/joint反例、K前缀/coverage/特权best-of-K，以及prior未来泄漏与真实机器人评价边界。
Stage3 S3.1–S3.8全部工程与学习完成，不代表大型视频模型复现或真实机器人能力。
Current Task=S3.8；Next Task=S4.1 — VLA Proposal，仅计划。STOP，不自动开始Stage4。
