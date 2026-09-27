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


def _lookup_nested(data: dict, dotted_or_slashed: str):
    """Return a value from either a flat 'a/b/c' key or nested dicts."""
    if dotted_or_slashed in data:
        return data[dotted_or_slashed]
    cur: object = data
    for part in dotted_or_slashed.replace(".", "/").split("/"):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _first_image(data: dict, *keys: str):
    for key in keys:
        value = _lookup_nested(data, key)
        if value is not None:
            return value
    return None


def make_a10_example(*, include_top: bool = False) -> dict:
    """Random A10 observation matching the current single-wrist training layout."""
    example = {
        "observation/state": np.random.rand(7).astype(np.float32),
        "observation/images/right": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "prompt": "pick_and_place",
    }
    if include_top:
        example["observation/images/top"] = np.random.randint(256, size=(3, 224, 224), dtype=np.uint8)
    return example


@dataclasses.dataclass(frozen=True)
class A10Inputs(transforms.DataTransformFn):
    """Transforms A10 dataset fields into the common pi model input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state_raw = data.get("observation/state")
        if state_raw is None:
            state_raw = _lookup_nested(data, "observation.state")
        if state_raw is None:
            raise KeyError("Missing observation/state")
        state_trajectory = np.asarray(state_raw, dtype=np.float32)

        if state_trajectory.ndim == 2:
            state = state_trajectory[0].copy()
        elif state_trajectory.ndim == 1:
            state = state_trajectory.copy()
        else:
            raise ValueError(
                f"Expected observation/state to have shape [T, 7] or [7], got {state_trajectory.shape}"
            )

        # Scene camera (optional). Single-cam A10 VR data only has the wrist view.
        primary_base_raw = _first_image(
            data,
            "observation/images/top",
            "observation/images/base_camera",
            "observation.images.top",
            "observation.images.base_camera",
        )
        # Wrist camera: dataset key is observation.images.right_wrist_0_rgb.
        right_raw = _first_image(
            data,
            "observation/images/right",
            "observation/images/right_wrist_0_rgb",
            "observation.images.right",
            "observation.images.right_wrist_0_rgb",
        )
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

