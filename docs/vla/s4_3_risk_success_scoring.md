# S4.3 — Risk / Success Scoring

## Problem

给S4.2冻结候选未来加上透明risk/success分数，并用独立校准/评价数据检查误报漏报。
本任务只输出分数和诊断flags，没有action selection、reject/fallback或闭环执行。
不把几何分数称为概率，也不把校准集表现当held-out证据。

## Intuition

先量出预测路径离禁入区域多近、末点离目标多远，再看真实执行是否同样如此。
模型有误差时，可以给判断加裕量：风险更容易报警，成功更难宣告。
裕量过大时“从不报成功”能做到零成功误报，却漏掉所有真实成功；需要正负样本分母和常量基线一起看。

## Input / Output Contract

[script](../../scripts/s4_3_risk_success_scoring.py)，[tests](../../tests/test_s4_3_risk_success_scoring.py)。
默认读取`tmp/s4_2_verified`，自己的产物用`--verification tmp/my_s4_2`。
要求S4.2 contract/report/model/verification及其原S4.1文件仍在；严格核对版本、plant语义、
模型与proposal hashes、动作/初态一致性、clock、AR重算及独立truth重放。缺失报错，不重训/下载。

| 项 | shape / dtype | 模态、单位、frame、时间与用途 |
|---|---|---|
| source actions | (16,K,3,2) float32 | world XY commanded displacement[m]；dt=.1s；原S4.1冻结bank |
| AR / actual truth | (16,K,4,2) float64 / float32 | world XY[m]，含初态+3步；**评分不使用TF预测** |
| RGB / text | (16,32,32,3) uint8 / (16,) Unicode | 仅取当下合成标记与literal语言，完美解析所选颜色目标 |
| goal | (G,2) probe float32 / VLA float64 | 当前已知world XY[m]，非从真实未来或成功标签反推 |
| risk / success score | (G,K) float64 | 有符号几何margin[m]；正值表示预测事件发生，不是概率 |
| actual margin / label | (G,K) float64 / bool | 从独立plant真值算，仅用于校准或事后评价 |
| flags | (G,K) bool | 有裕量的诊断判断，不控制执行或拒绝 |
| calibration contexts | G64/K5/H3 | initial/goal/action为float32；gain为float64，group_id为Unicode |
| held-out contexts | G32/K5/H3 | 独立seed与reset IDs；ID和gain.65条件共享这些contexts |
| buffers | 每事件scalar float64 | m，校准集group worst-error的q90 higher统计 |

输出`contract.json` schema `risk_success_margins/1.0`；`scores.npz`存raw probe输入、所有条件
predicted/actual trajectory、goal、score/margin/label/flag，禁用pickle读取；
`confusion.csv`含raw/calibrated/constant_negative/constant_positive的完整TP/FP/FN/TN；
`report.json`记录环境、源文件/源码hash、裕量、各分母和组覆盖统计。

VLA goal decoder是**单独的已知合成camera解析器**：从literal `move to red/blue`选RGB颜色，
9个彩色pixels的中心(u,v)映射world XY=((u−16)·.02,(16−v)·.02)m。
这是理想goal grounding假设，未修改或证明VLA视觉/语言能力；不读checkpoint内的evaluator goal。

## Core Concepts

- Risk只定义为：h1…h3任一**采样位置**落在闭圆盘内。圆心[.015,−.09]m，半径.02m。
  这是新增监测几何，不是墙、contact solver或plant约束，不改变ideal-point动力学。
- Success是：h3末位置到当前goal的L∞距离≤.01m。只评价.3s chunk终点，不是8步episode success。
- 两事件独立：到达goal且进入禁入区可以同时为真；没有把两者合成“安全成功”。
- 初态不参与未来risk判断；不检查相邻采样点连线与圆盘的连续碰撞。flags没有真实安全保证。
- 原VLA chunk全部不成功，只有这些负例无法评价success漏报；另建独立手工probe提供正负类。
  Probe不是VLA动作，probe成功不能冒充VLA改进。
- 误差裕量是经验校准，不是训练新head、sigmoid概率校准或正式coverage保证。

## Mathematics

给定predicted future p̂_h(h1…3)、已知goal g与禁入圆盘(c,r)：

risk margin R̂ = r − min_h ||p̂_h−c||₂。
success margin Ŝ = ε − ||p̂₃−g||∞，ε=.01m。

实际margin R,S用相同公式和actual plant positions计算；actual labels为R≥0、S≥0（边界计入）。
原始判断：R̂≥0报警；Ŝ≥0报成功。

对calibration reset group i的K5相关候选，分别算
E_i^risk=max_k |R̂_ik−R_ik|，E_i^success=max_k |Ŝ_ik−S_ik|。
B_risk/B_success取各自64个E的q90，NumPy `method='higher'`（升序第58项，1-based）。
裕量只来自calibration；heldout/VLA labels不参与选择buffer或阈值。

CLI α=`--margin-scale`，默认1，修改实验0：

- risk flag = [R̂ ≥ −α·B_risk]。
- success flag = [Ŝ ≥ α·B_success]。

α增加只会增加risk flags，减少success flags。固定真值集下：risk FN不增、FP不减；
success FP不增、FN不减。α=0退回raw判断，score/label/buffer均不变。
每个事件分别算TP/FP/FN/TN；FPR=FP/(FP+TN)，FNR=FN/(TP+FN)，不存在的类分母→null。
不能用总体accuracy或零FP掩盖无正例/全负预测。

额外报告`buffer_group_coverage`：每group的worst absolute margin error≤B的组数/总组数。
它衡量经验误差包络覆盖，不是事件概率可靠性，也不等于flag准确率。

## Architecture / Data Flow

冻结S4.2 model/futures + 当前goal → geometric scores → frozen calibrated buffers → diagnostic flags。
独立校准支路：64 synthetic contexts → fixed-model AR与独立plant truth → margins → q90 buffers。
独立heldout支路：32新contexts → 同样预测/执行 → labels → confusion；**不回流校准器**。
VLA支路：原S4.2 nominal/gain-shift futures → scores/labels → 80候选全分母报告。
拒绝的2条unknown输入仅保留记录，没有伪造score或被称为风险拦截。

## Math / Architecture → Code

`load_source`严格校验S4.2与S4.1来源并冻结model；`margins`只接trajectory和current known goal，
不接真实labels/gain。`probe_contexts`先生成initial/goal，再生成actions，不使用未来真值设计目标。
`calibrate`只读calibration margins，group max后取q90；`decisions`实现裕量方向。
`evaluate/confusion`保留全分母与null；`main`单独报告原始规则、校准规则与两常量基线，保留所有raw。

## Minimal Experiment

CPU repo .venv / NumPy，无新依赖/安装/数据下载。只读S4.2 model，无fit/重新训练。
Probe contexts仅小型合成数值：seed431生成64calibration groups，seed432生成32heldout groups。
K5、H3、initial x∈[−.08,.08]m/y∈[−.14,.02]m，goal为initial加±.08m随机offset。
每候选命令≈(goal−initial)/3乘独立[.4,1.6]因子，加8mm标准差噪声再clip±.06m。
ID plant gain按reset group独立均匀取[.8,1)，5候选共享该gain；额外heldout条件固定gain.65。
这些是有意提供正负样本的诊断probe，不是新的策略数据或真实robot demonstrations。

先读Problem→Mathematics，第二条先预测α1→0的不变量与误报漏报方向：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_3_risk_success_scoring.py --output tmp/my_s4_3
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/s4_3_risk_success_scoring.py --margin-scale 0 --output tmp/my_s4_3_raw
```

若使用自己的S4.2目录，两条均加同一`--verification tmp/my_s4_2`。
看contract.json、report.json、scores.npz、confusion.csv；尤其检查success all-negative失败案例。

## Expected Result

增加裕量会让risk更保守、success更难成立，不能保证总体更好。
Calibration与heldout必须隔离；在gain机制变化下，经验buffer覆盖可下降。
VLA负例组无法检验success召回，必须明确FNR=null，而不是0。

## Actual Result

2026-10-20，repo .venv Python3.12.3 / NumPy2.5.4 CPU；模块在本repo .venv，无安装/下载/训练。
默认risk buffer=5.476mm，success buffer=15.516mm。Calibration有64×5=320候选，
heldout ID/gain.65各32×5=160候选；VLA每条件16×5=80候选，但只有旧8布局×2文本，非16独立布局。
下表每格FP/FN，括号列真实正/负例分母；positive定义对每事件分别固定：

| 集合 | risk正/负 | risk raw FP/FN | risk α1 FP/FN | success正/负 | success raw FP/FN | success α1 FP/FN |
|---|---:|---:|---:|---:|---:|---:|
| calibration（非heldout） | 23/297 | 1/2 | 15/0 | 55/265 | 11/17 | 0/55 |
| heldout ID | 24/136 | 0/1 | 6/0 | 32/128 | 6/10 | 0/32 |
| heldout gain.65 | 23/137 | 2/2 | 7/0 | 42/118 | 23/37 | 0/42 |
| VLA nominal | 56/24 | 0/0 | 0/0 | 0/80 | 0/0 | 0/0 |
| VLA source gain.65 | 48/32 | 8/0 | 8/0 | 0/80 | 0/0 | 0/0 |

Success分数上界是ε=10mm，小于15.516mm buffer，因此α1**不可能报成功**，与constant_negative基线相同。
零FP伴随ID漏掉32/32、shift漏掉42/42真实成功；不能声称校准改善了success识别。
VLA success的0 FN来自正例0，不是完美召回；FNR正确输出null。

Risk/success group error-buffer覆盖：calibration均58/64；heldout ID为29/32与31/32；
heldout gain.65均8/32。Shift显著降低误差包络覆盖，但本次risk flags仍0FN；两指标并不等价。
VLA shifted coverage分别0/16和10/16 contexts，不能用probe覆盖率为VLA提供保证。

最终scoring batch80 median/P95约.0176/.0290ms，30测量/5warmup；仅geometry+flags，
不含prediction/goal decoding/plant/IO/sensor/通信/执行器，无整机实时结论。
6项新增测试、全部45项测试通过；scale0回归确认raw scores/labels/数据/source/buffers不变，
flags等于raw规则、混淆计数与常量基线全分母正确；精确几何公式/group quantile独立核对通过。
产物与准确命令见handoff；α0具体混淆值对应上表raw列。

## Result / Explanation

工程建立了透明的margin→threshold→diagnostic flag→heldout confusion链路，
并暴露成功裕量退化成全负预测、校准误差包络在机制偏移下失效的限制。
本节没有learned risk/success head或经过校准的概率，没有拒绝动作或改善闭环。
S4.1 candidate0闭环0/16的能力结论保留；本节三步goal occupancy标签不能替代它。

## Failure Analysis / Limitations

| 情况 | 影响/当前证据 |
|---|---|
| calibrated success全负 | FN全部真实成功；必须与constant_negative基线比较，不能只报0FP |
| 没有success正例的VLA bank | FNR=null，不能评价召回或声称全对 |
| group/candidate泄漏 | calibration/heldout用独立seeds/IDs，先group max，未按candidate随机拆分 |
| gain机制偏移 | heldout margin-error覆盖降至8/32；经验buffer非泛化保证 |
| perfect goal grounding | 合成RGB解析器是理想假设，不能归功于learned VLA |
| discrete risk监测 | 可能漏掉两采样点间穿圆盘；不检查初始风险、contact或真实碰撞 |
| 风险与success冲突 | 可同时为真，尚无多目标选择或安全执行准则 |
| probe非VLA / 重复旧布局 | 正例用于诊断，不证明VLA控制能力；candidate相关，不作统计稳健性声明 |
| no closed-loop | 没有rejection/fallback/argmin/replan，report closed_loop_success=null |

## Relation to Robotics

让planning/verification的几何判断与真实标签能被审计；真实系统还需不确定状态、连续碰撞、
控制器/延迟与风险代价定义。监测圆盘不改变plant或action语义；sibling repo未修改或导入。

## Interview Capsule / Notes

30秒：我用future禁入圆盘margin和末点goal margin评分，64个独立reset组校准误差裕量，
32个heldout组测FP/FN。Risk漏报下降但误报增加；success裕量大于容差导致全负预测，零FP不是成功。

2分钟：解释为什么margin不是概率、为何候选要按group校准、VLA全负组为什么FNR=null、
为什么probe成功不能算VLA成功，以及gain shift下包络coverage与flag准确率的区别。

## Likely Follow-up Questions

1. risk/success的几何公式、单位、事件边界、时间范围是什么？为什么能同时为真，为什么不能称连续碰撞检测？
2. scorer读取什么，goal从哪里来？score/flag/actual label的shape与dtype是什么？TF和真实未来何时可以使用？
3. calibration、heldout与原VLA候选各有多少groups/candidates？为什么先group max再取q90，哪些数据不参与拟合buffer？
4. α1→0哪些量不变，两个事件的FP/FN如何变化？为什么15.516mm success buffer会退化成全负预测？
5. VLA success正例0时FNR为何null？group error-buffer coverage、概率校准、闭环success和局部latency各不能互相替代什么？

## Must Remember

有符号margin≠概率；group校准；保留正负分母；零FP可能全负预测；geometry flag≠动作拒绝或机器人成功。

## My Verification

- [x] Run：检查两个margin、独立calibration/heldout、VLA正例0与success全负基线。
- [x] Modify：先预测α1→0的数据/score/buffer不变量及FP/FN取舍，再运行新目录核对。
- [x] Explain：独立回答上述五问，手算320/160/80分母与全负预测的FP/FN。

Engineering Complete + Learning Mastered（2026-10-20 用户确认）。Current Task=S4.3；Next Task=S4.4 — Rejection / Alternate Action仅计划，STOP。

2026-10-20 用户明确确认实验和预测均已完成，五项Explain准确说明离散未来risk圆盘与末点success的
公式、m单位、含边界和独立事件；当前RGB/literal goal来源、AR/truth与score/flag/label shape/dtype，
TF不用于评分、真实未来只供校准或评价；64/32独立reset组与旧8布局的分母，group max后q90 higher第58项，
heldout/VLA不参与拟合。准确解释alpha1→0不变量和FP/FN方向及具体计数、success buffer大于score上界
造成constant_negative退化；正确区分正例0时FNR=null、group覆盖与概率校准/分类准确率、flags与闭环执行、
局部latency与整机实时。Run/Modify/Explain依据用户报告全部勾选；不增加风险拦截或闭环能力声明。
