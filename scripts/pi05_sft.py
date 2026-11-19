"""Run RLinf statistics, inspect the first worker batch, or launch single-GPU SFT."""
import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "pi05_head_rgb_sft"
EXTENSION = "real_world_rl.integrations.rlinf.pi05_head_rgb"


def check_batch(cfg):
    import torch
    from openpi.shared import normalize
    from rlinf.data.datasets.openpi import build_openpi_sft_dataloader

    stats_file = Path(cfg.actor.model.openpi_data.norm_stats_path)
    expected = normalize.load(stats_file.parent)
    loader, data = build_openpi_sft_dataloader(cfg, 1, 0, cfg.data.train_data_paths)
    import numpy as np

    for key in ("actions", "state"):
        for field in ("mean", "std", "q01", "q99"):
            values = getattr(expected[key], field)
            if values.shape != (32,) or not np.isfinite(values).all():
                raise ValueError(f"invalid {key}.{field} in {stats_file}")
            np.testing.assert_array_equal(values, getattr(data.norm_stats[key], field))
    obs, actions = next(iter(loader))
    n = cfg.actor.micro_batch_size
    horizon = cfg.actor.model.openpi.action_horizon
    if obs.state.shape != (n, 32) or actions.shape != (n, horizon, 32):
        raise ValueError("unexpected state or action shape")
    if not torch.isfinite(actions).all() or not torch.isfinite(obs.state).all():
        raise ValueError("non-finite normalized actions or states")
    for name, active in (("base_0_rgb", True), ("left_wrist_0_rgb", False), ("right_wrist_0_rgb", False)):
        if obs.images[name].shape != (n, 3, 224, 224) or not torch.all(obs.image_masks[name] == active):
            raise ValueError(f"invalid image or mask for {name}")
    if obs.tokenized_prompt.shape != (n, cfg.actor.model.openpi.max_token_len):
        raise ValueError("invalid token shape")
    counts = obs.tokenized_prompt_mask.sum(dim=1)
    if not (counts > 0).all():
        raise ValueError("empty tokenized prompt")
    print(json.dumps({"norm_stats": str(stats_file), "actions": list(actions.shape),
                      "state": list(obs.state.shape), "rgb": list(obs.images["base_0_rgb"].shape),
                      "prompt": list(obs.tokenized_prompt.shape), "valid_tokens": counts.tolist(),
                      "image_masks": {k: v.tolist() for k, v in obs.image_masks.items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stats", "batch", "train"))
    parser.add_argument("--rlinf", type=Path, required=True, help="RLinf source checkout")
    parser.add_argument("--dataset", type=Path, required=True, help="export containing meta/info.json")
    parser.add_argument("--run-dir", type=Path, required=True, help="stats and training output")
    parser.add_argument("--model", type=Path, help="RLinf-format pi05 base checkpoint (train)")
    parser.add_argument("--override", action="append", default=[], help="Hydra override for batch/train")
    args = parser.parse_args()
    dataset, rlinf, run = (p.resolve() for p in (args.dataset, args.rlinf, args.run_dir))
    entry = rlinf / "examples/sft/train_vla_sft.py"
    if not entry.is_file() or not (dataset / "meta/info.json").is_file():
        parser.error("check --rlinf and --dataset")
    if args.stage == "train" and (args.model is None or not args.model.is_dir()):
        parser.error("train requires --model pointing to an RLinf-format pi05 checkpoint")
    if args.stage == "stats" and args.override:
        parser.error("--override applies to the Hydra batch/train stages")
    model = args.model.resolve() if args.model else run
    run.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_LEROBOT_HOME=str(dataset.parent), PI05_REPO_ID=dataset.name,
                      PI05_RUN_DIR=str(run), PI05_MODEL_PATH=str(model),
                      EMBODIED_PATH=str(entry.parent), RLINF_EXT_MODULE=EXTENSION)
    roots = [str(ROOT), str(rlinf)]
    sys.path[:0] = roots
    os.environ["PYTHONPATH"] = os.pathsep.join(roots + [os.environ.get("PYTHONPATH", "")])
    os.chdir(run)
    config_dir = ROOT / "configs/rlinf"
    if args.stage == "train":
        os.execv(sys.executable, [sys.executable, str(entry), "--config-path", str(config_dir),
                                  "--config-name", CONFIG, *args.override])

    from real_world_rl.integrations.rlinf.pi05_head_rgb import register
    register()
    if args.stage == "stats":
        from toolkits.lerobot.calculate_norm_stats import main as calculate
        calculate("pi05_head_rgb", dataset.name)
        return

    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=CONFIG, overrides=args.override)
    check_batch(cfg)


if __name__ == "__main__":
    main()
