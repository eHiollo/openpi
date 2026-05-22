import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class A10Inputs(transforms.DataTransformFn):
    """Transforms A10 dataset fields into the common pi model input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state_trajectory = np.asarray(data["observation/state"], dtype=np.float32)

        if state_trajectory.ndim == 2:
            state = state_trajectory[0].copy()
        elif state_trajectory.ndim == 1:
            state = state_trajectory.copy()
        else:
            raise ValueError(
                f"Expected observation/state to have shape [T, 7] or [7], got {state_trajectory.shape}"
            )

        # Base / scene RGB: training used `observation.images.top`; clients may send
        # `observation/images/base_camera` instead (same model slot: base_0_rgb).
        top_raw = data.get("observation/images/top")
        base_camera_raw = data.get("observation/images/base_camera")
        if top_raw is not None:
            primary_base_raw = top_raw
        else:
            primary_base_raw = base_camera_raw
        right_raw = data.get("observation/images/right")
        ref_raw = primary_base_raw if primary_base_raw is not None else right_raw
        if ref_raw is None:
            ref_hwc = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            ref_hwc = _parse_image(ref_raw)

        top_hwc = (
            _parse_image(primary_base_raw) if primary_base_raw is not None else np.zeros_like(ref_hwc)
        )
        right_hwc = _parse_image(right_raw) if right_raw is not None else np.zeros_like(ref_hwc)
        left_hwc = np.zeros_like(ref_hwc)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (top_hwc, left_hwc, right_hwc)
                image_masks = (primary_base_raw is not None, np.False_, right_raw is not None)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (top_hwc, left_hwc, right_hwc)
                image_masks = (primary_base_raw is not None, np.False_, right_raw is not None)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # 'actions' is already prepared by RepackTransform in config:
        # - either from observation.state
        # - or from dataset action
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)

            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(
                    f"Expected actions to have shape [T, 7], got {actions.shape}"
                )

            inputs["actions"] = actions.copy()

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs

@dataclasses.dataclass(frozen=True)
class A10Outputs(transforms.DataTransformFn):
    """Returns only valid A10 action dimensions."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)

        if actions.ndim != 2:
            raise ValueError(f"Expected actions to have shape [T, 7+], got {actions.shape}")

        return {"actions": actions[:, :7]}

