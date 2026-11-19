# 从 SpaceMouse 真机人工示教到 RLinf π₀.₅ 关节 SFT

现场顺序是 **检查设备和动作来源 → 人工录一条抓放 → 验收首条 → 批量录制 → 导出数据集 → 计算统计量 → 检查首个 batch → 训练**。录制时由人操作机械臂和夹爪，无需启动 VLA。下面的命令从 `real-world-rl` 根目录运行。格式、采集频率与官方示例见[真机数据采集说明](rlinf-pi05-real-data.zh-CN.md)。

## 1. 先确认工位能提供关节训练标签

| 每个控制步要采什么 | 来源和用途 |
|---|---|
| 七轴绝对关节目标、夹爪命令 | 取遥操链实际下发的目标。导出后前七列按 q1…q7 排列，单位为弧度，第八列是夹爪命令。这八列用于监督训练。 |
| 实际关节角、实际夹爪开度、实际 TCP 位姿 | 取 follower 的反馈，保留动作前后观测。关节角和开度作为模型状态。TCP 用米和 xyzw 四元数保存，供对齐检查与后续使用。 |
| 遥操末端命令 | SpaceMouse 路径保留发出的笛卡尔控制量。如控制器产生绝对 TCP 目标，一并记录并注明参考系。 |
| 头部 RGB、时间和回合信息 | 将头图与本步动作、反馈对应，保存时间、人工接管状态及成功标记，导出时附任务文字。只需实拍头图。 |

**目前只有 SpaceMouse 时，先完成第 2、3 节的一条真机联调，再决定能否批量采集。** 当前 tracker 发六维笛卡尔增量，记录实测关节角和 TCP，档案里缺少实际下发的七轴绝对关节目标。要把这一路接入关节训练，需让控制端暴露与控制步对应的关节目标，或接入实际向机械臂发送关节目标的遥操控制器，并把该目标接入录制和导出。首条要逐步对照控制端发送记录。下一帧实测关节角、离线逆解和修改 YAML 模式名都无法还原当时下发的命令。

**工位若有关节 leader，可直接从第 4 节开始，用当前代码走到训练。** 两种设备只需选一条采集路径。原始数据保留绝对值，本次按绝对关节目标训练；若后续需要 delta，可根据训练方定义在同一回合内计算。

## 2. 接入 SpaceMouse 并核对方向

将 SpaceMouse 的 USB 接在运行 `rwrl teleop run` 的 Linux 主机上。在本仓库根目录、已激活的采集环境中执行以下命令。station 应选用已调通的机械臂、头部相机和 SpaceMouse profile。相机名字与预设的 `obs.rgb` 保持一致。

```bash
python -m pip install -e '.[spacemouse]'
mkdir -p local
cp -n examples/presets/spacemouse_head_rgb.yaml local/spacemouse_head_rgb.yaml
rwrl --station local/station.yaml discover
rwrl --station local/station.yaml camera show --name head --frames 30
rwrl --station local/station.yaml teleop profiles list
rwrl --station local/station.yaml --json teleop diagnose --duration-s 10
```

诊断只读输入。分别轻推、侧移、提压和扭转旋钮，再逐个按下两枚按钮，对照报告确认六轴方向及哪个实体按钮对应 `button_0`、`button_1`。首次配置或换操作方位，先按工位的实际 task frame 标定并选中 profile，再重复诊断。[SpaceMouse 标定与 USB 排查](spacemouse-commissioning.md)

## 3. 用 SpaceMouse 人工录一条抓放

核对现场 home、工作空间和夹爪，清空复位路径。执行命令并查看终端预览，确认后机械臂可能回 home 并打开夹爪。首次只录一条。

```bash
rwrl --station local/station.yaml --preset local/spacemouse_head_rgb.yaml teleop run \
  --mode tracker --record local/spacemouse-pick-001.pkl --num-successes 1 --max-episodes 1
```

终端进入录制后按下表操作。

| 顺序 | 手上或键盘的操作 | 现场效果 |
|---|---|---|
| 准备 | 放开旋钮和两枚按钮，等待终端就绪 | 通过中立检测，机械臂保持当前位置 |
| 接管 | 旋钮居中时，单击按钮 0，然后松开 | 进入遥操，再按一次按钮 0 会交还控制 |
| 靠近 | 轻推、侧移或提压旋钮控制末端平移，倾斜或扭转控制姿态 | 方向依标定 profile。偏转越大，目标移动越快。旋钮回中后目标停止推进，手臂可能继续跟到已发目标 |
| 抓取 | 在接管中单击按钮 1 | 夹爪从打开切换到闭合 |
| 放置 | 操作旋钮移动到落点，回中，再单击按钮 1 | 夹爪从闭合切换到打开 |
| 成功 | 保持接管，先让旋钮回中，再按终端键盘 **s** | 本回合标为成功并进入保持 |
| 退出 | 以上单条命令完成后按 **q** | 释放控制，检查同名 pkl、contract.json 和 trajectory.sqlite3 |

中途暂停并交还控制可按按钮 0，恢复时再按一次。失误按键盘 `f` 标为失败，`r` 丢弃重录。多条采集时调高命令中的回合额度，每回合结束后摆好物体，按 `n`，释放旋钮和所有按钮，等待中立确认并开始下一回合。额度耗尽后按 `q` 退出。[本地按键说明](teleop.md#spacemouse) · [回合保持与退出](spacemouse-commissioning.md#supervised-recording-sessions)

## 4. 用已支持的关节 leader 取得首条关节示教

以下步骤使用关节 leader 提供七轴绝对目标，导出命令的输入是这一路生成的 `local/pick-001.pkl`。

### 配置工位与预设

从 real-world-rl 根目录运行。预设假设已配置单台七轴 Franka、支持开度反馈的夹爪、名为 head 的头部相机，并由同一 robot daemon 提供带夹爪通道的关节 leader（例如 GELLO）。若 GELLO 接在采集机本地，使用仓库已有的 `--teleop-device gello --teleop-options-yaml <标定文件>`。[安装说明](tutorial/01-install.md) · [关节遥操作配置](gello.md)

```bash
mkdir -p local
cp -n examples/presets/pi05_head_rgb.yaml local/pi05_head_rgb.yaml
rwrl --station local/station.yaml workspace show
rwrl --station local/station.yaml camera show --name head --frames 30
```

先把工位的七组关节限位按 q1…q7 顺序填入副本的 `real_robot.joint_limits`，每组为弧度 [low, high]。`joint_abs` 用这些值将命令编码和解码。然后将 `real_robot.remote` 指向已启动的 daemon，核对七轴顺序、home、夹爪和头部相机名称。将预设顶层与 real_robot 下的 control_hz 均设为 10，并检查头图能拍全抓取和放置过程。若相机名为 front，同时改动 `obs.rgb` 和导出参数 `--camera front`。本仓库的关节模式需要使用已调试的关节控制工位，启用了笛卡尔 task workspace 的 station 会拒绝关节轨迹。reset 会执行工位配置的 home 并打开夹爪。

### 录制一次抓放

```bash
rwrl supervise -- rwrl record --config local/pi05_head_rgb.yaml \
  --output-path local/pick-001.pkl --num-successes 1 --max-episodes 1
```

将 leader 对齐 follower，在终端按 t 接管，用 leader 移动手臂和夹爪。完成任务后，在接管状态按 s 标记成功。f 标记失败，r 丢弃重录，h 交还控制，q 结束会话。上例最多执行一次尝试。批量采集可调整两个 episode 参数，连续回合之间会 reset，使用新的输出路径。原始 `.trajectory.sqlite3` 保存全部尝试，成功示教另存 `.pkl` 和同名 `.contract.json`。[录制器说明](recording-demos.md)

## 5. 验收第一条真机数据

先看同名 contract。以下以第 4 节的成功记录为例。若将路径改为第 3 节的默认 SpaceMouse 档案，该检查会在动作模式处停止，此时回到第 1 节补齐关节命令来源。若改造后的控制器单独记录关节目标，须先适配导出器读取该字段。

```bash
python - <<'PYTHON'
from pathlib import Path
import json
c = json.loads(Path("local/pick-001.contract.json").read_text())
print(c["action_mode"], c["arm_action_dim"], c["state_fields"])
assert c["action_mode"] == "joint_abs" and c["arm_action_dim"] == 7
assert c["gripper_in_action"]
assert c["state_fields"][:3] == ["joint_pos", "gripper_pos", "tcp_pose"]
PYTHON
```

### 试导出一条

将可信的同名 pkl 和 contract.json 放在一起，在独立的导出环境中执行。多个同任务、同配置的 pkl 可以一次传入。

```bash
python -m pip install 'lerobot==0.3.3' 'huggingface-hub<1'
python examples/export_pi05_head_rgb.py local/pick-001.pkl \
  --output local/first_check/pick_block --task 'Pick up the block and place it in the tray.'
```

### 核对首条的动作、反馈和图像

查看导出时打印的 actual 和 max gap，核对夹爪开合时段有无长停顿。读取首帧，确认 actions 和 state 各 8 维、tcp_pose 为 7 维，关节以弧度表示，头图和任务文字对应本次操作。

```bash
python - <<'PYTHON'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("pick_block", root="local/first_check/pick_block")
row = ds[0]
print(ds.num_episodes, len(ds), ds.fps, row["task"])
for key in ("actions", "env_action", "state", "next_state", "tcp_pose", "next_tcp_pose"):
    print(key, ds.features[key]["names"], row[key].tolist())
print("image", tuple(row["image"].shape))
PYTHON
```

再抽查靠近、闭爪、抬起和释放附近的帧，对照头图、夹爪命令、实际开度和 TCP 运动方向。确认动作前七列为 `target_q1_rad` 至 `target_q7_rad`。10 Hz 的相邻控制步预期间隔约 0.1 秒，重点检查抓取或释放瞬间是否缺帧。

## 6. 批量录制并交付训练集

沿用通过首条验收的设备、控制模式、头图和频率。以下为关节 leader 示例，目标为再录 10 次成功示教，最多尝试 20 次；可按现场计划调整。使用新文件名，每次复位后重新摆好物体。

```bash
rwrl supervise -- rwrl record --config local/pi05_head_rgb.yaml \
  --output-path local/pick-session-002.pkl --num-successes 10 --max-episodes 20
```

在导出环境中，将首条和后续同任务的成功档案一起导出到新的 LeRobot v2.1 目录。无需中转 HDF5。保留原始档案和对应 contract。

```bash
python examples/export_pi05_head_rgb.py local/pick-001.pkl local/pick-session-002.pkl \
  --output datasets/pick_block --task 'Pick up the block and place it in the tray.'
```

核对完成后，将整个 `datasets/pick_block/` 复制到训练机，保留 data、meta 及其中的控制配置和时间记录。

## 7. 在训练机计算统计量、检查 batch 并启动训练

使用官方 RLinf OpenPI 环境。以下配方对应 RLinf 提交 `a29c0a6c37b02881b8d4af0900fa97f35e44b34c` 和 `rlinf-openpi==0.1.1`。在 GPU 训练机按[官方环境与 π₀.₅ 权重转换说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_openpi.html)安装依赖。第一次构造模型输入时 OpenPI 会读取 PaliGemma tokenizer，需要缓存或下载该文件。

从本仓库根目录在 Bash 中运行。把 `--dataset` 指到包含 `meta/info.json` 的目录，并将 --rlinf 与 --model 换成训练机上的实际路径。

```bash
common=(--rlinf /path/to/RLinf --dataset datasets/pick_block --run-dir local/pick-sft)
python scripts/pi05_sft.py stats "${common[@]}"
test -s local/pick-sft/assets/pi05_head_rgb/pick_block/norm_stats.json
python scripts/pi05_sft.py batch "${common[@]}"
```

确认首个 batch 的头图掩码为 true、两个腕部掩码为 false，动作张量形状为 [1, 10, 32]。在同一 Bash 会话中启动训练。

```bash
python scripts/pi05_sft.py train "${common[@]}" --model /path/to/pi05_base_openpi
```
