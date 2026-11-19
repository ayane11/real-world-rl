"""Export successful rwrl record pickles as local single-camera LeRobot v2.1 episodes."""
import argparse
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np


JOINTS = [f"q{i}_rad" for i in range(1, 8)]
STATE = [*JOINTS, "gripper_width_fraction"]
TCP = ["x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"]


EXTRA_FEEDBACK = {
    "joint_vel": [f"dq{i}_rad_s" for i in range(1, 8)],
    "tcp_vel": ["vx_m_s", "vy_m_s", "vz_m_s", "wx_rad_s", "wy_rad_s", "wz_rad_s"],
    "tcp_force": ["fx_N", "fy_N", "fz_N"],
    "tcp_torque": ["tx_Nm", "ty_Nm", "tz_Nm"],
}


def read_contract(path):
    contract = json.loads(path.with_suffix(".contract.json").read_text())
    experiment = contract["experiment"]
    if (contract["action_mode"] not in ("joint_delta", "joint_abs") or not contract["gripper_in_action"]
            or contract.get("num_arms", 1) != 1 or contract["arm_action_dim"] != 7
            or contract["state_fields"][:3] != ["joint_pos", "gripper_pos", "tcp_pose"]
            or any(k not in EXTRA_FEEDBACK for k in contract["state_fields"][3:])
            or experiment["observation"]["convert_obs_rotation"]):
        raise ValueError(f"{path}: expected seven joint commands, joint positions, gripper and TCP pose")
    if contract["action_mode"] == "joint_abs":
        limits = np.asarray(contract["joint_limits"], dtype=np.float32)
        if limits.shape != (7, 2) or not np.isfinite(limits).all() or not (limits[:, 0] < limits[:, 1]).all():
            raise ValueError(f"{path}: joint_abs requires seven valid joint limits in radians")
    elif not np.isfinite(contract["joint_scale_rad"]) or contract["joint_scale_rad"] <= 0:
        raise ValueError(f"{path}: joint_delta requires a positive joint_scale_rad")
    hz = experiment["control_hz"]
    if hz <= 0 or int(hz) != hz:
        raise ValueError(f"{path}: LeRobot needs positive integer fps, got {hz}")
    return contract


def frame(row, camera, contract):
    obs = row["obs"]
    image = np.asarray(obs[f"rgb_{camera}"])
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("expected one HWC uint8 RGB image")
    fields = {"image": image}
    for observation, prefix in ((obs, ""), (row["next_obs"], "next_")):
        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        extras = contract["state_fields"][3:]
        size = 15 + sum(len(EXTRA_FEEDBACK[key]) for key in extras)
        if state.shape != (size,) or not np.isfinite(state).all():
            raise ValueError(f"expected {size} finite feedback values in {prefix}state")
        if abs(np.linalg.norm(state[11:15]) - 1) > 0.01 or not 0 <= state[7] <= 1:
            raise ValueError("invalid TCP quaternion or gripper width")
        fields[prefix + "state"] = state[:8]
        fields[prefix + "tcp_pose"] = state[8:15]
        offset = 15
        for key in extras:
            width = len(EXTRA_FEEDBACK[key])
            fields[prefix + key] = state[offset:offset + width]
            offset += width
    action = np.asarray(row["action"], dtype=np.float32).reshape(-1)
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError("expected seven joint commands and one gripper command")
    if np.abs(action).max() > 1.00001:
        raise ValueError("expected normalized env action")
    commands = action.copy()
    if contract["action_mode"] == "joint_delta":
        commands[:7] *= contract["joint_scale_rad"]
    else:
        low, high = np.asarray(contract["joint_limits"], dtype=np.float32).T
        commands[:7] = low + (commands[:7] + 1) * (high - low) / 2
    fields["actions"] = commands
    fields["env_action"] = action
    return fields


def segments(rows):
    segment = []
    for row in rows:
        valid = np.asarray(row["human_action_mask"]).all() and not row["syncing"]
        valid = valid and not row.get("teleop_events", {}).get("teleop_syncing", False)
        if segment:
            prev = segment[-1]
            if (not valid or prev.get("training_segment_end") or prev["episode_id"] != row["episode_id"]
                    or prev["step_index"] + 1 != row["step_index"]):
                yield segment
                segment = []
        if valid:
            segment.append(row)
    if segment:
        yield segment


def export(paths, output, camera, task, writer):
    if output.exists():
        raise FileExistsError(output)
    if not task.strip():
        raise ValueError("supply the instruction for these demonstrations")
    contract = read_contract(paths[0])
    fps = int(contract["experiment"]["control_hz"])
    dataset = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        root = Path(tmp) / "dataset"
        for path in paths:
            if read_contract(path) != contract:
                raise ValueError(f"{path}: recording contracts differ")
            with path.open("rb") as stream:
                rows = pickle.load(stream)  # Only load trusted rwrl recordings.
            for segment in segments(rows):
                times = np.array([row["timestamp_s"] for row in segment])
                gaps = np.diff(times)
                if not np.isfinite(times).all() or (gaps <= 0).any():
                    raise ValueError(f"{path}: invalid recorder timestamps")
                for row in segment:
                    fields = frame(row, camera, contract)
                    if dataset is None:
                        prefix = "delta_q" if contract["action_mode"] == "joint_delta" else "target_q"
                        action_names = [f"{prefix}{i}_rad" for i in range(1, 8)] + ["gripper_cmd"]
                        features = {
                            "image": {"dtype": "image", "shape": fields["image"].shape,
                                      "names": ["height", "width", "channel"]},
                            "env_action": {"dtype": "float32", "shape": (8,),
                                           "names": [f"joint{i}_norm" for i in range(1, 8)] + ["gripper_cmd"]},
                            "actions": {"dtype": "float32", "shape": (8,), "names": action_names},
                        }
                        feedback = {"state": STATE, "tcp_pose": TCP}
                        feedback.update({k: EXTRA_FEEDBACK[k] for k in contract["state_fields"][3:]})
                        for key, names in feedback.items():
                            for prefix in ("", "next_"):
                                features[prefix + key] = {"dtype": "float32", "shape": (len(names),),
                                                          "names": names}
                        dataset = writer.create(repo_id=output.name, root=root, fps=fps,
                                                features=features, use_videos=False)
                    dataset.add_frame(fields, task=task)
                dataset.save_episode()
                # Keep the recorder clock separately from LeRobot's regular training grid.
                with (root / "meta" / "rwrl_segments.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps({
                        "episode_index": dataset.meta.total_episodes - 1,
                        "recording": path.name, "source_episode": segment[0]["episode_id"],
                        "first_step": segment[0]["step_index"], "step_end_unix_s": times.tolist(),
                    }) + "\n")
                timing = (f"actual={1/gaps.mean():.2f} Hz, max gap={gaps.max():.3f}s"
                          if gaps.size else "single frame")
                print(f"{path.name} {segment[0]['episode_id']} step {segment[0]['step_index']}: "
                      f"{len(segment)} frames, {timing}", flush=True)
        if dataset is None:
            raise ValueError("no contiguous fully human-labelled frames")
        (root / "meta" / "rwrl_contract.json").write_text(json.dumps(contract, indent=2))
        root.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demos", type=Path, nargs="+", help="trusted rwrl success .pkl files")
    parser.add_argument("--output", type=Path, required=True, help="new dataset directory")
    parser.add_argument("--task", required=True, help="shared language instruction")
    parser.add_argument("--camera", default="head", help="station camera name")
    args = parser.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    export(args.demos, args.output, args.camera, args.task, LeRobotDataset)


if __name__ == "__main__":
    main()
