# P4 roadmap

每行是用户 0.5～2 小时的阅读/运行/修改/解释任务，不是自动执行队列。
Current Task S4.6（Engineering Complete，Learning pending）；Next Task无自动后续任务，等待S4.6学习确认与用户选择。S0.1 Engineering Complete + Learning Mastered（2026-10-13 用户确认）；S0.2–S0.7 Engineering Complete + Learning Mastered（2026-10-14 用户确认）；Stage0全部工程与学习完成，Stage1的S1.1工程与学习完成（2026-10-17 用户确认），S1.2工程与学习完成（2026-10-17 用户确认），S1.3工程与学习完成（2026-10-17 用户确认），S1.4工程与学习完成（2026-10-17 用户确认），S1.5工程完成（闭环失败）、Learning Mastered（2026-10-18 用户确认），S1.6工程与学习完成（2026-10-18 用户确认），S1.7工程与学习完成（2026-10-18 用户确认），Stage1工程与学习包全部完成但模型控制仍失败；Stage2的S2.1工程与学习完成，S2.2工程与学习完成，S2.3工程与学习完成，S2.4工程与学习完成，S2.5工程与学习完成；S2.6 Engineering Complete + Learning Mastered（2026-10-19用户确认）；S2.7 Engineering Complete + Learning Mastered（2026-10-19用户确认）；Stage2全部工程与学习完成；S3.1 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S3.2 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S3.3 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S3.4 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S3.5 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S3.6 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S3.7 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S3.8 Engineering Complete + Learning Mastered（2026-10-20用户确认）；Stage3全部工程与学习完成；Stage4 S4.1 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S4.2 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S4.3 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S4.4 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S4.5 Engineering Complete + Learning Mastered（2026-10-20用户确认）；S4.6 Engineering Complete，Learning pending；Stage0–4全部工程完成，策略能力仍未达标。
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
| S1.5 Tiny Transformer VLA | 2 | [学习包](vla/s1_5_tiny_transformer_vla.md)：16平均image patches/3word/proprio tokens→Transformer→H3delta；masked loss/E-prefix/oracle与完整失败评价；Engineering Complete（闭环失败）+ Learning Mastered |
| S1.6 Language Generalization | 1 | [学习包](vla/s1_6_language_generalization.md)：train-only word selector与lookup；独立新句/词类OOV/颜色/语序对照，oracle bridge不冒充VLA；Engineering Complete + Learning Mastered |
| S1.7 Closed-loop VLA Rollout | 1.5 | [学习包](vla/s1_7_closed_loop_vla_rollout.md)：冻结checkpoint与协议，expert-state/visited-state误差、失败stage/全分母与分阶段latency；Engineering Complete + Learning Mastered |

## Stage 2 — Modern VLA Architecture Study

不平均投入；重点 OpenVLA 与 π0。Paper→Architecture→Code Reading→Data-flow→Tensor Shape→Tiny Reimplementation。
每个任务查官方论文/代码，记录日期、版本/commit、精确变体；不能把其他版本特点混入原模型。
完整版硬件/数据成本解释在 docs，当前仅 CPU conceptual version。

| Task | h | 目标与可检查产物 |
|---|---:|---|
| S2.1 RT-1 / RT-2 Overview | 1 | [学习包](vla/s2_1_rt_overview.md)：原版架构/固定源码与codec词表实验；Engineering Complete + Learning Mastered |
| S2.2 Octo Study | 1.5 | [学习包](vla/s2_2_octo_study.md)：固定源码conditioning/readout/diffusion head与NumPy mask/代数验证；Engineering Complete + Learning Mastered |
| S2.3 OpenVLA Architecture Trace | 2 | [学习包](vla/s2_3_openvla_architecture_trace.md)：原版image/text、双视觉feature融合/BOS序列、动作codec与unnorm统计；Engineering Complete + Learning Mastered |
| S2.4 FAST / π0-FAST Action Representation | 1.5 | [学习包](vla/s2_4_fast_action_representation.md)：DCT/量化/train-only整数BPE、heldout压缩重建与bins对照；Engineering Complete + Learning Mastered |
| S2.5 π0 Architecture Trace | 2 | [学习包](vla/s2_5_pi0_architecture_trace.md)：原版expert/state/flow、不同宽度attention与cache、ODE数学对照；Engineering Complete + Learning Mastered |
| S2.6 π0.5 Study | 1 | [学习包](vla/s2_6_pi05_study.md)：论文训练/层级与开源state/time分支对照；bin、hybrid隔离mask、adaRMS探针；Engineering Complete + Learning Mastered |
| S2.7 VLA Architecture Comparison | 1.5 | [学习包](vla/s2_7_vla_architecture_comparison.md)：五策略版本/模态/动作/数据/适配统一表；E/K与假设部署预算实验；Engineering Complete + Learning Mastered |

## Stage 3 — Robot World Models

禁止直接从大型 video world model 开始；每个 learned model 有 held-out evaluation。

| Task | h | 目标与可检查产物 |
|---|---:|---|
| S3.1 State Dynamics | 1.5 | [学习包](vla/s3_1_state_dynamics.md)：226参数residual MLP、episode held-out、物理误差与不动/理想加法基线；Engineering Complete + Learning Mastered |
| S3.2 Multi-step Rollout | 1 | [学习包](vla/s3_2_multistep_rollout.md)：冻结checkpoint，1/5/20/100步TF与AR、偏置探针与完整漂移/分布变化评价；Engineering Complete + Learning Mastered |
| S3.3 Model Uncertainty | 1.5 | [学习包](vla/s3_3_model_uncertainty.md)：episode-bootstrap小ensemble、独立名义score阈值、ID/OOD与低分歧高误差反例；Engineering Complete + Learning Mastered |
| S3.4 Latent State Representation | 2 | [学习包](vla/s3_4_latent_state.md)：train-only PCA观测压缩、latent MLP、d2/d1重建/预测/AR与aliasing；Engineering Complete + Learning Mastered |
| S3.5 Reward / Success / Contact Prediction | 1.5 | [学习包](vla/s3_5_outcome_heads.md)：一步reward/success/contact/failure heads，独立标签定义、FP/FN与动作OOD/阈值对照；Engineering Complete + Learning Mastered |
| S3.6 Model-based Planning | 1.5 | [学习包](vla/s3_6_model_based_planning.md)：嵌套候选→learned rollout→显式cost→select→独立toy plant执行，coverage/regret/失败分期；Engineering Complete + Learning Mastered |
| S3.7 MPC-style Replanning | 1.5 | [学习包](vla/s3_7_mpc_replanning.md)：冻结模型、真实反馈/无新状态/open-loop对照、E1/4/10与gain/kick；Engineering Complete + Learning Mastered |
| S3.8 Video / Generative World Model Concepts | 1.5 | [学习包](vla/s3_8_video_world_model_concepts.md)：固定DreamerV3/GAIA-1/Genie2架构，解析action/stochastic joint与像素/物理反例；Engineering Complete + Learning Mastered |

## Stage 4 — VLA × World Model Integration

CPU Embodied Agent Mini-System：Language Goal→Tiny VLA→Candidate Chunk→World Model→
Future/Success/Risk→Accept/Reject→Execute Short Prefix→Observe→Replan。

| Task | h | 目标与可检查产物 |
|---|---:|---|
| S4.1 VLA Proposal | 1 | [学习包](vla/s4_1_vla_proposal.md)：冻结tiny VLA、原始+手工bias chunks、版本化action contract；Engineering Complete + Learning Mastered（2026-10-20用户确认） |
| S4.2 World-model Verification | 1 | [学习包](vla/s4_2_world_model_verification.md)：匹配ideal plant的冻结线性残差模型、候选AR/TF与独立实际未来、gain shift；Engineering Complete + Learning Mastered（2026-10-20用户确认） |
| S4.3 Risk / Success Scoring | 1 | [学习包](vla/s4_3_risk_success_scoring.md)：几何margins、独立group误差校准/heldout FP/FN、success全负反例；Engineering Complete + Learning Mastered（2026-10-20用户确认） |
| S4.4 Rejection / Alternate Action | 1 | [学习包](vla/s4_4_rejection_alternate.md)：冻结gate、备选/zero fallback、bad/good拒绝计数与一次H3执行；Engineering Complete + Learning Mastered（2026-10-20用户确认） |
| S4.5 Receding-horizon Replanning | 1.5 | [学习包](vla/s4_5_receding_horizon.md)：冻结VLA/WM/gate、真实/预测state重规划、E1/3/kick与全分母latency；Engineering Complete + Learning Mastered（2026-10-20用户确认） |
| S4.6 Final Benchmark | 2 | [学习包](vla/s4_6_final_benchmark.md)：新16pairs/两protocol seeds、matched VLA-only/WM、全分母risk/success/rejection/latency与配对反例；Engineering Complete，Learning pending |

超过两小时则先把当前任务拆为小包，仍逐任务等待用户选择，不扩张当前授权。
