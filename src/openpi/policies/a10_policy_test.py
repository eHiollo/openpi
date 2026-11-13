import numpy as np
import pytest

from openpi import transforms
from openpi.models import model
from openpi.policies import a10_policy
from openpi.training import config


def _sample(dim=14):
    return {
        "observation/state": np.arange(dim, dtype=np.float32),
        "actions": np.arange(30 * dim, dtype=np.float32).reshape(30, dim),
        **{
            f"observation/images/{key}": np.full((3, 8, 8), value, dtype=np.uint8)
            for key, value in (("top", 10), ("left", 20), ("right", 30))
        },
    }


def test_dual_camera_mapping_and_output():
    result = a10_policy.A10Inputs(model.ModelType.PI05, action_dim=14)(_sample())
    assert result["state"].shape == (14,)
    assert result["actions"].shape == (30, 14)
    assert all(result["image_mask"].values())
    for key, value in (("base_0_rgb", 10), ("left_wrist_0_rgb", 20), ("right_wrist_0_rgb", 30)):
        np.testing.assert_array_equal(result["image"][key], np.full((8, 8, 3), value))
    padded = np.ones((30, 32), dtype=np.float32)
    assert a10_policy.A10Outputs(action_dim=14)({"actions": padded})["actions"].shape == (30, 14)


def test_dual_config_delta_roundtrip_and_prompt():
    cfg = config.get_config("pi05_a10_dual_finetune")
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    sample = _sample()
    original = sample["actions"].copy()
    result = transforms.compose(data.data_transforms.inputs)(sample)
    joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    np.testing.assert_array_equal(result["actions"][:, joints], original[:, joints] - np.arange(14)[joints])
    np.testing.assert_array_equal(result["actions"][:, [6, 13]], original[:, [6, 13]])
    restored = transforms.compose(data.data_transforms.outputs)(result)
    np.testing.assert_allclose(restored["actions"], original)
    prompted = transforms.InjectDefaultPrompt(cfg.data.default_prompt)({})
    assert prompted["prompt"] == "Fold the garment neatly using both arms."
    assert not data.prompt_from_task
    assert data.action_sequence_keys == ("observation.state",)


def test_single_arm_regression():
    sample = _sample(7)
    del sample["observation/images/top"]
    result = a10_policy.A10Inputs(model.ModelType.PI05)(sample)
    assert result["image_mask"] == {"base_0_rgb": False, "left_wrist_0_rgb": False, "right_wrist_0_rgb": True}
    assert a10_policy.A10Outputs()({"actions": np.ones((30, 32))})["actions"].shape == (30, 7)
    trajectory = _sample(7)
    trajectory["observation/state"] = np.ones((30, 7))
    assert a10_policy.A10Inputs(model.ModelType.PI05)(trajectory)["state"].shape == (7,)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observation/state", np.zeros(7)),
        ("observation/state", np.zeros((0, 14))),
        ("actions", np.zeros((30, 7))),
    ],
)
def test_dual_rejects_wrong_dimensions(field, value):
    sample = _sample()
    sample[field] = value
    with pytest.raises(ValueError, match="Expected"):
        a10_policy.A10Inputs(model.ModelType.PI05, action_dim=14)(sample)


@pytest.mark.parametrize("camera", ["top", "left", "right"])
def test_dual_requires_all_cameras(camera):
    sample = _sample()
    del sample[f"observation/images/{camera}"]
    with pytest.raises(KeyError, match="requires"):
        a10_policy.A10Inputs(model.ModelType.PI05, action_dim=14)(sample)


def test_output_rejects_short_action():
    with pytest.raises(ValueError, match="Expected"):
        a10_policy.A10Outputs(action_dim=14)({"actions": np.zeros((30, 7))})
