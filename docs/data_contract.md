# Episode file contract — embodied_episode/1.0

## Boundary and versioning

每个 episode 用同 stem 的 `episode.npz` 和 `episode.json`；不依赖旧 repo 的 Python。
`schema_version = "embodied_episode/1.0"`，本地 reader 拒绝其他版本与未声明数组。
破坏时序/语义/字段的修改升级版本，adapter 明确转换；不猜测或静默迁移。
第一版只支持非空、固定 control_dt、同步采样 episode。不规则时钟需要未来新契约。

## Temporal contract

T = 已执行的控制转移数，不是观测数。t = 0,…,T−1：
先采集 observation_t，再选择并执行 action_t 于 [timestamp_t,timestamp_{t+1})，
最后采集 observation_{t+1}。最终 observation_T 保留，无 action_T。
Episode language 在本版本内不变；指令变化应另开 episode 或设计新版本。
`terminated[t]` / `truncated[t]` 描述 action_t 执行后的边界。
仅最后一行允许边界，恰有一种为 true；自然结束与时间截断不同，success 独立。

| NPZ field | shape | dtype | modality / meaning |
|---|---|---|---|
| timestamp | (T+1,) | float64 | episode-relative seconds, starts at 0, stride control_dt |
| proprioception | (T+1,P) | float32 | 因果 robot measurements；逐维字段/单位/frame 定义 |
| action | (T,A) | float32 | 已执行控制命令；不能直接假定是 qpos / ctrl |
| terminated | (T,) | bool | task/environment naturally ended after transition |
| truncated | (T,) | bool | time/resource limit after transition |
| next_state (optional) | (T,P) | float32 | 本版本仅为 proprioception[1:] 的显式副本 |
| image (optional) | (T+1,H,W,3) | uint8 | RGB HWC，0–255；camera frame/calibration/source 必须声明 |
| force (optional) | (T+1,F) | float32 | 各通道 physical quantity，显式 N / N·m 和 frame |
| tactile (optional) | (T+1,K) | float32 | 扁平有序 taxel/channels，列出单位/frame/位置语义 |

所有数值必须有限；没有 padding。可选模态缺失是 absent，不是 zero；稀疏、异步传感不能伪造同步。
同步说明必须记录；真实传感 latency 应在生产者侧显式处理，不把未来测量插入 o_t。

## JSON metadata

Required: schema_version、episode_id、task_name、embodiment、language_instruction、success(bool)、
control_dt(seconds)、action_representation(object)、modalities(object)、provenance(object)、success_definition(string)。
provenance 至少含 source / source_schema / source_commit / controller / sampling / group_id；未知值写 `unknown`，不编造。

modalities 中每个 NPZ 数组有 shape、dtype、semantic、temporal 字段；vector 另有长度匹配的
fields 列表，每项 name / unit / frame。timestamp 单位 s，flags 单位 1；image 有 frame/color_space。
这些声明由 validator 检查，但传感器是否真的因果、单位是否正确仍需 exporter 审计。

action_representation 至少含 type、anchor、execution、normalized(false)。
可选类型 joint_absolute / joint_delta / cartesian_absolute / cartesian_delta；每列的名称、unit、frame 在 action.fields。
Joint delta 必须说明锚定 measured q_t 还是 previous reference；Cartesian delta 必须说明平移/旋转约定和坐标系。
Action 不等于速度：delta[m] 只有除以明确 dt 才可解释为平均速度。
真实 absolute/delta reference 通常经 servo 执行，不保证 measured next_state = reference。
不同 embodiment 的关节数、顺序、limit、tool frame 和 controller 不同，不能只补齐维度就统一。

## S0.1 toy instance

2D ideal point actuator；P=A=2，fields x/y [m], world frame。
语言 `move right`；action type cartesian_delta，anchor measured position at t，
execution `ideal displacement applied once per control interval`，control_dt=0.1 s。

```
timestamp: [0.0, 0.1, 0.2, 0.3]
proprioception: [[0,0], [0.01,0], [0.02,0], [0.03,0]]
action: [[0.01,0], [0.01,0], [0.01,0]]
terminated: [false,false,true]; truncated: [false,false,false]
```

success = final x ≥ 0.03m；这里只是显式 toy score。`move left` 对应负方向和 final x ≤ −0.03m。
配对 `(proprioception[t], language_instruction) → action[t]`；这是示例标签映射，不是已训练 VLA。
Action chunk 是 action[t:t+H]，不能跨 episode；S0.2 已实现state-only sequence的padding/mask，见 [学习包](vla/s0_2_sequence_dataset.md)；S0.7已增加synthetic RGB/language context与rule chunk/prefix实验，仍未训练foundation policy。

## Old repository export plan (not implemented)

| Source | Observed contract | Required conversion |
|---|---|---|
| S19.1 episode_contract.py | observations(T+1,6), actions(T,2), time_s; world XY absolute reference, 50Hz / physics1kHz | float32 state/action，重命名，保留 goal fields；action cartesian_absolute + feedback controller；manifest 分为每 episode JSON |
| S22.2 demonstration_recorder.py | observations(T+1,10), actions(T,2), times; absolute slide servo reference; scripted master | 复制数值文件语义，明确 force/touch 和 held_ctrl；quality eligibility 不冒充 task success；补语言和边界 |
| S22.6 dexterous_bc.py / dexterous_task.py | 50Hz 因果状态、past ctrl/phase，6维 absolute joint reference [rad] | 明确观测字段与 joint order/servo，保留 reset group；避免把 S22.5 1kHz audit proxy 误当因果 policy observation |

旧 manifest 没有自然语言，不能声称其原本带有 human language annotations；人工补充需记录来源。
不运行或修改旧 repo exporter，本轮只是设计边界。S0.3已实现 [合成RLDS概念adapter](vla/s0_3_rlds_oxe_concepts.md)：canonical loaded steps的最后action无效，最后observation的is_terminal映射至内部最后transition。实际RLDS/TFDS/OXE reader尚未实现，内部不绑定RLDS；缺有效terminal observation、success、dt或单位时不编造。
原始数据不归一化；训练统计仅从 train episode 拟合并独立存储，验证/test 不参与。

## S0.4 observed real-source gap (no conversion)

[真实样本检查](vla/s0_4_real_robot_dataset_inspection.md)读取一条Bridge社区LeRobot转存，28行state与28行action，
一相机28帧。该表不是canonical RLDS loaded-step输入；不能套用toy规则自动丢弃末action。
缺明确有效终点、terminated/truncated、独立success及动作单位/frame/controller/anchor，producer映射也未知。
存储clock/视频PTS匹配只证明记录时间对应。当前只保存inspection report与preview，不生成embodied_episode/1.0。
本契约未扩展、未放宽；不补造信息，不将inspection检查通过等同语义适配通过。

## S0.5 action views (raw contract unchanged)

[连续动作表示实验](vla/s0_5_continuous_action_representation.md)创建synthetic two-joint servo的joint_absolute raw episodes。
动作r_t为rad reference，经显式q_next=q+gain*(r−q)执行；T4/T+1=5，末budget truncated、success独立。
derived measured-anchor delta=r_t−q_t；previous-reference delta=r_t−r_(t−1)，显式r_-1=q_0。
Cartesian views为已知planar FK下的XY目标/目标差值[m]；不提供唯一逆映射或可执行Cartesian controller。
五view各有独立train-only normalizer sidecar，normalized值不覆盖raw NPZ，不改变normalized=false规则。
chunk复用S0.2，按mask处理真实槽；future measured-anchor delta不能总加chunk起点测量。
这些语义只属于本toy；S0.4未知producer样本不因此变得可转换。未升级或放宽embodied_episode/1.0。

## S0.6 token views (raw contract unchanged)

[离散动作实验](vla/s0_6_discrete_action_tokenization.md)以S0.5 joint_absolute raw rad为输入，
另存scalar_action_bins/1.0 codec与int64 IDs/float32 decoded views，raw物理文件不改。
range只fit原始train actions；内部edge归右、端点范围内、中心解码，超range显式clip+saturation或reject。
IDs依赖channel namespace及精确B/range/rules与unit/frame/anchor/controller/dt；不是可直接执行的角度或通用机器人vocabulary。
decoded reference只用于已知toy counterfactual replay，不替换原始记录action或success；没有升级raw契约或转换真实Bridge。

## S0.7 RGB/language chunk context (raw contract unchanged)

[动作chunk学习包](vla/s0_7_action_chunk_representation.md)保留point XY[m]的T+1状态、RGB及float64 clock，
按reset group先split；S0.2窗口附加精确anchor RGB，不让future/terminal image或success进入planner。
derived target/predicted为(N,H,2)float32 world displacement[m]，bool mask按t+k<T决定，与zero数值无关。
H prediction与E execution prefix分开，mask仅用于offline标签有效性，不传入rule planner。
32×32 camera明确pixel-world calibration；真实sensor同步未因toy验证而解决；无learned VLA或真实robot success声明。
使用已有optional image契约，不改变embodied_episode/1.0，不放宽S0.2 state-only reader或S0.4 source条件。
