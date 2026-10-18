# P4 roadmap

每行是用户 0.5～2 小时的阅读/运行/修改/解释任务，不是自动执行队列。
Current Task S1.5（Engineering Complete，闭环失败、Learning待确认）；Next Task S1.6（仅计划）。S0.1 Engineering Complete + Learning Mastered（2026-10-13 用户确认）；S0.2–S0.7 Engineering Complete + Learning Mastered（2026-10-14 用户确认）；Stage0全部工程与学习完成，Stage1的S1.1工程与学习完成（2026-10-17 用户确认），S1.2工程与学习完成（2026-10-17 用户确认），S1.3工程与学习完成（2026-10-17 用户确认），S1.4工程与学习完成（2026-10-17 用户确认），S1.5工程完成（失败实验）、学习待确认，S1.6及后续任务 Planned。
所有任务最终都有 Engineering Code/Experiment/Docs 与 Learning Run/Modify/Explain 双状态。
不重学 P0–P3；只在新语境中引用既有知识。

## Stage 0 — Embodied AI Data & Action Representation

| Task | h | 目标与可检查产物 |
|---|---:|---|
| S0.1 VLA Data Contract | 1 | observation/language/action/episode/trajectory/embodiment；三步 NPZ+JSON 和时间对齐验证 |
| S0.2 Episode / Sequence Dataset | 1.5 | [学习包](vla/s0_2_sequence_dataset.md)：toy history/horizon、padding/mask；episode/reset-group split，跨界与泄漏反例 |
| S0.3 RLDS / Open X-Embodiment Concepts | 1 | [学习包](vla/s0_3_rlds_oxe_concepts.md)：官方 schema study 与 tiny metadata，合成 adapter 字段映射；不下载全集 |
| S0.4 Real Robot Dataset Inspection | 1.5 | [学习包](vla/s0_4_real_robot_dataset_inspection.md)：一条Bridge社区LeRobot转存，固定1.93MB，CPU读数值/图像、审计clock与契约差异；不声称原始RLDS已验证 |
| S0.5 Continuous Action Representation | 1 | [学习包](vla/s0_5_continuous_action_representation.md)：servo reference与两种delta anchor、Cartesian views、train-only normalization、chunk及embodiment mismatch |
| S0.6 Discrete Action Tokenization | 1.5 | [学习包](vla/s0_6_discrete_action_tokenization.md)：train-range bins→channel IDs→center decode，half-bin界/边界/saturation与toy replay；不复制大型tokenizer |
| S0.7 Action Chunk Representation | 1 | [学习包](vla/s0_7_action_chunk_representation.md)：causal state/RGB/language→rule future chunk；复用mask/index，区分H/E并比较扰动toy rollouts |

## Stage 1 — Tiny Vision-Language-Action Policy

| Task | h | 目标与可检查产物 |
|---|---:|---|
| S1.1 Language-conditioned State Policy | 1.5 | [学习包](vla/s1_1_language_conditioned_state_policy.md)：NumPy MLP / 无语言基线；reset-held-out物理误差与完整toy闭环；Learning Mastered |
| S1.2 Tiny Language Representation | 1 | [学习包](vla/s1_2_tiny_language_representation.md)：train-only tokens / learned embedding / masked mean，PAD与UNK诊断、known/OOV闭环；Engineering Complete + Learning Mastered |
| S1.3 Tiny Visual Encoder | 1.5 | [学习包](vla/s1_3_tiny_visual_encoder.md)：32×32 RGB→116参数CNN空间features/heatmap；held-out定位、GAP与遮挡/blank/oracle toy闭环；Engineering Complete + Learning Mastered |
| S1.4 Multimodal Fusion | 1.5 | [学习包](vla/s1_4_multimodal_fusion.md)：冻结视觉summary + literal条件 + state，concat/pre-LN Transformer；三模态消融、shape trace与完整toy闭环；Engineering Complete + Learning Mastered |
| S1.5 Tiny Transformer VLA | 2 | [学习包](vla/s1_5_tiny_transformer_vla.md)：16平均image patches/3word/proprio tokens→Transformer→H3delta；masked loss/E-prefix/oracle与完整失败评价；Learning待确认 |
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
