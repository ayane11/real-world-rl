# RLinf 真机 π₀.₅ SFT：采集格式、频率和字段

**这个工位可以只用头部 RGB。** 团队训练的是关节策略，建议每个控制步同时记录图像、遥操发出的绝对关节目标和机械臂实测反馈（含关节角、末端 TCP 位姿和夹爪开度）。只有末端位姿和 RGB 时，训练端缺少操作者发出的关节命令。本仓库的执行命令见[真机采集到训练操作手册](pi05-head-rgb.zh-CN.md)。

## 官方训练入口接收什么格式？

RLinf 的 OpenPI 自定义数据集流程读取 **LeRobot 数据集目录**，其中 `meta/info.json` 描述字段和维度，`data/` 保存逐帧数据。官方 Franka 示例在遥操作时标记成功回合、填写任务描述，设置 `env.data_collection.export_format: lerobot_dataset` 即可导出。该单臂示例演示的是 π₀。[Franka 采集与导出](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/franka_pi0_sft_deploy.html) · [OpenPI 自定义数据集](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_openpi.html)

更接近本项目的是官方 **YAM 真机 π₀.₅ 关节 SFT**。遥操作时同时记录绝对关节位置和目标动作，按回合写成 LeRobot 分片，训练前合并成功示教，再用 `pi05_yam_joint` 计算状态与动作的归一化统计。YAM 的 14 维向量来自两条六轴臂及两个夹爪。本项目使用一条七轴臂和一个夹爪，应按本机维度注册数据适配器。[YAM 真机采集](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/yam.html) · [YAM π₀.₅ 训练](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/yam_openpi.html)

**采集端无需先写 HDF5。** 可以直接写 LeRobot，也可以沿用工位的原始档案，离线转换。已有 HDF5 的团队可按 episode 将观测、控制命令、时间和任务说明转换为上述训练输入。本仓库保留 SQLite 全量档案与成功示教 pkl，再导出 LeRobot v2.1，RGB 随 Parquet 保存。[本地导出实现](../examples/export_pi05_head_rgb.py)

**遥操设备怎么选？** 官方单臂 Franka 示例用 SpaceMouse 控制末端。你们现有的 SpaceMouse 可以走本仓库的 tracker 录制末端命令和跟随反馈。若训练标签要求“发给机械臂的绝对关节目标”，需要从控制器记录该目标，或采用能提供七轴目标的关节 leader。仅靠七轴实测反馈无法还原同一步的遥操目标。[官方 Franka 采集说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/franka_pi0_sft_deploy.html) · [本地 SpaceMouse 操作步骤](pi05-head-rgb.zh-CN.md)

## 每秒要采几次？

RLinf 的 OpenPI SFT 指南没有给所有机器人规定统一的 Hz。官方 YAM 的部署配置以 30 Hz 执行动作，预测窗口为 30 步。这个数字对应 YAM 的动作时间尺度。本工位的慢速抓放预设选择 **10 Hz**，每 100 ms 配一组观测与遥操命令，10 步预测窗口约覆盖 1 秒。[OpenPI SFT 指南](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_openpi.html) · [YAM 动作频率与窗口](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/yam_openpi.html) · [本地预设](../examples/presets/pi05_head_rgb.yaml)

头部相机可以按 30 fps 出图，采集器按控制步读取图像。动作前的 RGB 和状态对应本步命令，动作后的反馈另存。LeRobot 的规则时间轴是 `frame_index / fps`，本仓库还在 `meta/rwrl_segments.jsonl` 保存每步结束的主机时间。可用首条真机示教的实际频率和最大步间隔检查夹爪开合、相机传输等环节的延迟。当前配对精度是控制步级，部署时应使用与采集一致的动作周期。

## 只有头部相机，怎么接入？

官方单臂 `pi0_realworld` 适配器读取两路图像，检查 19 维状态和 7 维动作。如果数据只有头图，直接套用这个配置会遇到字段或维度不匹配。本项目将头图映射到 `base_0_rgb`，两个腕部槽位在模型输入中填零并设 `image_mask=false`，磁盘只保存真实头图。固定相机时要让夹爪、抓取物和落点在整段动作中可见，尤其要检查闭爪时是否被机械臂遮挡。[官方字段映射](https://github.com/RLinf/RLinf/blob/main/rlinf/models/embodiment/openpi/dataconfig/realworld_dataconfig.py) · [官方输入维度与掩码](https://github.com/RLinf/RLinf/blob/main/rlinf/models/embodiment/openpi/policies/realworld_policy.py) · [本地单相机适配器](../real_world_rl/integrations/rlinf/pi05_head_rgb.py)

## 本工位每步采哪些数据？

| LeRobot 字段 | 含义 | 默认训练用途 |
|---|---|---|
| image | 动作前头部 RGB，uint8，320×240 | 唯一有效视角 base_0_rgb，两路腕部图像 mask 为 false |
| state | [q1,…,q7,gripper_width_fraction]，关节为实测弧度，夹爪为实测开度比例（0 闭、1 开） | 8 维状态 |
| tcp_pose | 与 state 同一控制步的实测 [x,y,z,qx,qy,qz,qw]，位置单位米，姿态是 xyzw 四元数，沿用机械臂后端坐标系 | 保留为单独 7 维列 |
| joint_vel / tcp_vel | 实测关节速度（rad/s）、末端线速度和角速度（m/s、rad/s） | 跟随分析 |
| tcp_force / tcp_torque | Franka 后端反馈的末端外力与外力矩估计（N、N·m） | 接触分析 |
| next_state / next_tcp_pose / 其他 next_ 反馈列 | 同一条 transition 执行后的对应实测值，含片段最后一步 | 跟随分析与离线做差 |
| env_action | 录制器保存的 8 维归一化环境命令 | 核对命令映射 |
| actions | 和当前观测配对的七轴绝对目标角（rad），以及 -1 闭、+1 开的夹爪命令。关节标签来自该步提交给环境的示教动作，设备限幅和跟踪在下游 | 8 维监督目标 |
| task / task_index | 该次成功示教的真实语言指令 | 构造 prompt |
| episode_index / frame_index / timestamp | 成功回合内连续、全维度人工接管的片段各自成为一个 episode | 构造动作块，避免跨越交还控制或重新同步的间隔 |

原始 pkl 的 `obs.state` 和 `next_obs.state` 均按预设顺序保存 34 个数值：关节角 7、夹爪开度 1、TCP 位姿 7、关节速度 7、TCP 速度 6、外力 3、力矩 3。导出后反馈按字段命名，动作前后各保留一份。默认 SFT 只读取 image、state、actions 和任务指令。导出只保存当前 RGB，单帧 320×240×3 的未压缩 RGB 约 225 KiB，同步保存两组 34 维 float32 反馈仅 272 字节。

## 绝对关节量怎样派生 delta？

记 `u[t] = actions[:7]` 为本步遥操目标，`q[t] = state[:7]` 为动作前实测角，`q_after[t] = next_state[:7]` 为动作后实测角。三者均按 q1…q7 排列，单位 rad。本项目默认直接用绝对目标训练。

| 训练或分析用途 | 计算方法（七个关节分别计算） |
|---|---|
| 相对当前实测姿态的动作 | `u[t] - q[t]`。若整个预测窗口都以当前观测为锚点，第 k 个标签为 `u[t+k] - q[t]` |
| 相邻遥操目标的变化 | `u[t] - u[t-1]`。只在同一连续 episode 内做差，首步没有前一个目标 |
| 机器人实际走过的关节位移 | `q_after[t] - q[t]`。可用于跟踪分析 |
| 一步后的目标跟踪误差 | `u[t] - q_after[t]`。动作后的反馈是该步采样值，机器人可能仍在运动 |

夹爪单独保留开合命令（-1 闭、+1 开）和实测开度（0 闭、1 开）。差分仅作用于关节维。训练和部署使用相同的 delta 定义。改成 delta 监督时，在训练输入变换中生成对应标签，并用该变换重新计算 norm_stats。绝对目标模型部署到本仓库时，用现场限位把反归一化后的弧度目标映射回环境的 [-1,1]。
