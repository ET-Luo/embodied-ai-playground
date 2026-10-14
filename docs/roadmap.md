# P4 roadmap

每行是用户 0.5～2 小时的阅读/运行/修改/解释任务，不是自动执行队列。
Current Task S0.3；Next Task S0.4。S0.1 Engineering Complete + Learning Mastered（2026-10-13 用户确认）；S0.2 Engineering Complete + Learning Mastered（2026-10-14 用户确认）；S0.3 Engineering Complete（2026-10-14），Learning未确认；S0.4及后续任务 Planned。
所有任务最终都有 Engineering Code/Experiment/Docs 与 Learning Run/Modify/Explain 双状态。
不重学 P0–P3；只在新语境中引用既有知识。

## Stage 0 — Embodied AI Data & Action Representation

| Task | h | 目标与可检查产物 |
|---|---:|---|
| S0.1 VLA Data Contract | 1 | observation/language/action/episode/trajectory/embodiment；三步 NPZ+JSON 和时间对齐验证 |
| S0.2 Episode / Sequence Dataset | 1.5 | [学习包](vla/s0_2_sequence_dataset.md)：toy history/horizon、padding/mask；episode/reset-group split，跨界与泄漏反例 |
| S0.3 RLDS / Open X-Embodiment Concepts | 1 | [学习包](vla/s0_3_rlds_oxe_concepts.md)：官方 schema study 与 tiny metadata，合成 adapter 字段映射；不下载全集 |
| S0.4 Real Robot Dataset Inspection | 1.5 | 网络适合时少量公开 DROID/Open-X style 样本；先报告大小/用途/必要性；读取、可视化、对比契约；不可得则明确未验证 |
| S0.5 Continuous Action Representation | 1 | absolute joint/joint delta/Cartesian delta/normalized/chunk 对比与 embodiment mismatch |
| S0.6 Discrete Action Tokenization | 1.5 | toy bins→IDs→decode，重建误差/边界/饱和；不复制大型 tokenizer |
| S0.7 Action Chunk Representation | 1 | vision/language context→future sequence 的表示；复用 P3 chunk 知识，验证末端 mask |

## Stage 1 — Tiny Vision-Language-Action Policy

| Task | h | 目标与可检查产物 |
|---|---:|---|
| S1.1 Language-conditioned State Policy | 1.5 | 小 MLP，same state/different instruction→different behavior；训练与 held-out 评价 |
| S1.2 Tiny Language Representation | 1 | tokenization/embedding/sentence representation；tiny learned embedding 优先，未知词诊断 |
| S1.3 Tiny Visual Encoder | 1.5 | 32×32/64×64 RGB→tiny CNN feature/token；可见物体实验与 held-out 评价 |
| S1.4 Multimodal Fusion | 1.5 | concatenate 与 token Transformer 对比；模态消融、shape trace |
| S1.5 Tiny Transformer VLA | 2 | visual/language/proprio tokens→Transformer→action chunk；CPU 小模型与评价 |
| S1.6 Language Generalization | 1 | pick red cube/move blue object 与 grab red block/move blue item；独立 paraphrase 测试，不声称大模型语义 |
| S1.7 Closed-loop VLA Rollout | 1.5 | success/action error/failure stage/inference latency；不仅报告 offline loss |

## Stage 2 — Modern VLA Architecture Study

不平均投入；重点 OpenVLA 与 π0。Paper→Architecture→Code Reading→Data-flow→Tensor Shape→Tiny Reimplementation。
每个任务查官方论文/代码，记录日期、版本/commit、精确变体；不能把其他版本特点混入原模型。
完整版硬件/数据成本解释在 docs，当前仅 CPU conceptual version。

| Task | h | 目标与可检查产物 |
|---|---:|---|
| S2.1 RT-1 / RT-2 Overview | 1 | modality/action token 与预训练迁移；架构比较，不完整训练 |
| S2.2 Octo Study | 1.5 | conditioning/action head/embodiment adaptation；tiny 数据流示意与评价 |
| S2.3 OpenVLA Architecture Trace | 2 | image+language(+robot context where applicable)→action；逐 module/tensor/shape/semantic，指出 context 的真实进入位置 |
| S2.4 FAST / π0-FAST Action Representation | 1.5 | 压缩/离散 action chunk 与 autoregression；toy tokenizer 对照，不完整复现 |
| S2.5 π0 Architecture Trace | 2 | foundation backbone+continuous flow/action chunk；shape trace；与 P3 Diffusion 的联系/区别 |
| S2.6 π0.5 Study | 1 | 与选定 π0/π0-FAST 版本比较；数据/泛化主张与本实验区分 |
| S2.7 VLA Architecture Comparison | 1.5 | OpenVLA/FAST/Octo/π0 family统一表：modality、fusion、action、continuous/discrete、AR/generative、chunk、data、adaptation、fine-tune、latency/deployment |

## Stage 3 — Robot World Models

禁止直接从大型 video world model 开始；每个 learned model 有 held-out evaluation。

| Task | h | 目标与可检查产物 |
|---|---:|---|
| S3.1 State Dynamics | 1.5 | (state_t,action_t)→next_state 小 MLP，baseline 与 held-out error |
| S3.2 Multi-step Rollout | 1 | one/5/20/long-horizon drift，区分 teacher-forcing 与 autoregressive rollout |
| S3.3 Model Uncertainty | 1.5 | 小 ensemble 的 ID/OOD disagreement；不能等同 calibrated risk |
| S3.4 Latent State Representation | 2 | observation→encoder→z；(z_t,a_t)→z_t+1，重建/预测评价 |
| S3.5 Reward / Success / Contact Prediction | 1.5 | success/reward/contact/failure-risk 小 heads，独立标签与错误类型 |
| S3.6 Model-based Planning | 1.5 | candidate actions→predicted rollout→score→select，真实环境验证 |
| S3.7 MPC-style Replanning | 1.5 | observe→plan→execute short prefix→replan；与 open-loop 对比 |
| S3.8 Video / Generative World Model Concepts | 1.5 | paper/architecture/tiny concept：action conditioning、temporal/physical consistency、latent prediction、stochastic futures；不训练大视频模型 |

## Stage 4 — VLA × World Model Integration

CPU Embodied Agent Mini-System：Language Goal→Tiny VLA→Candidate Chunk→World Model→
Future/Success/Risk→Accept/Reject→Execute Short Prefix→Observe→Replan。

| Task | h | 目标与可检查产物 |
|---|---:|---|
| S4.1 VLA Proposal | 1 | 冻结小策略生成候选 chunks，检查 action contract |
| S4.2 World-model Verification | 1 | 预测候选未来与实际未来对照 |
| S4.3 Risk / Success Scoring | 1 | 透明评分与 held-out 校准/误报漏报 |
| S4.4 Rejection / Alternate Action | 1 | reject + alternative/fallback，统计 rejected bad actions 与误拒好动作 |
| S4.5 Receding-horizon Replanning | 1.5 | execute prefix/observe/replan，审计真实闭环和 latency |
| S4.6 Final Benchmark | 2 | matched VLA only vs VLA+WM；success/failure/prediction error/rejected bad actions/inference+planning latency，报告种子/分母/限制 |

超过两小时则先把当前任务拆为小包，仍逐任务等待用户选择，不扩张当前授权。
