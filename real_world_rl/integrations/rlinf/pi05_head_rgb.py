"""OpenPI data config for examples/export_pi05_head_rgb.py (training host only)."""
import dataclasses

import numpy as np
from openpi import transforms
from openpi.training.config import DataConfigFactory, ModelTransformFactory


@dataclasses.dataclass(frozen=True)
class HeadRGBInputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data):
        image = np.asarray(data["image"])
        # LeRobot's image transform returns CHW floats in [0, 1].
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if np.issubdtype(image.dtype, np.floating):
            image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("expected a single RGB image")
        state = np.asarray(data["state"])
        if state.shape != (8,):
            raise ValueError(f"expected 8-D state, got {state.shape}")
        empty = np.zeros_like(image)
        result = {
            "image": {"base_0_rgb": image, "left_wrist_0_rgb": empty, "right_wrist_0_rgb": empty},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_,
                           "right_wrist_0_rgb": np.False_},
            "state": transforms.pad_to_dim(state, self.action_dim),
            "prompt": data["prompt"],
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim != 2 or actions.shape[-1] != 8:
                raise ValueError(f"expected an action chunk of shape (T, 8), got {actions.shape}")
            result["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        return result


@dataclasses.dataclass(frozen=True)
class JointOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :8]}


@dataclasses.dataclass(frozen=True)
class HeadRGBDataConfig(DataConfigFactory):
    def create(self, assets_dirs, model_config):
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            action_sequence_keys=("actions",),
            data_transforms=transforms.Group(inputs=[HeadRGBInputs(model_config.action_dim)],
                                             outputs=[JointOutputs()]),
            model_transforms=ModelTransformFactory()(model_config),
        )


def register():
    """RLINF_EXT_MODULE hook, also used by the standalone statistics and batch tools."""
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import DataConfig, TrainConfig
    from rlinf.models.embodiment.openpi import dataconfig

    name = "pi05_head_rgb"
    config = TrainConfig(
        name=name,
        model=Pi0Config(pi05=True, action_horizon=10, discrete_state_input=True),
        data=HeadRGBDataConfig(repo_id="pick_block", base_config=DataConfig(prompt_from_task=True)),
        # calculate_norm_stats drops the last partial batch. Include every recorded frame.
        batch_size=1,
        num_workers=0,
    )
    dataconfig._CONFIGS_DICT[name] = config
