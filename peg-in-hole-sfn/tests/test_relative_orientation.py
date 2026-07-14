from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sfn.evaluation.evaluate_perception import _load_model
from sfn.models import VirtualSensorNetwork
from sfn.models.orientation import OrientationNet, RelativeOrientationNet
from sfn.training.common import make_checkpoint, save_checkpoint
from sfn.training.perception import _model_for_task


def _valid_encoded_mask(batch: int = 2) -> torch.Tensor:
    x = torch.zeros(batch, 1, 64, 64)
    x[:, :, 12:52, 10:54] = 1.0  # visible hole/seam
    x[:, :, 22:44, 24:40] = 0.5  # peg occludes part of it
    return x


def test_relative_model_uses_shared_encoder_and_pair_only_fusion():
    model = RelativeOrientationNet(base=4).eval()
    assert sum(1 for module in model.modules() if module is model.encoder) == 1
    x = _valid_encoded_mask(1)
    pair = model.pair_features(x)
    assert pair.shape == (1, 32, 4, 4)

    # Hold the peg fixed and alter only seam geometry.  Both the correlation
    # representation and final scores must react to the hole side of the pair.
    changed = x.clone()
    changed[:, :, 12:22, 10:54] = 0.0
    with torch.no_grad():
        assert not torch.equal(model.pair_features(x), model.pair_features(changed))
        assert not torch.equal(model(x), model(changed))


def test_relative_training_config_round_trips_through_all_loaders(tmp_path: Path):
    model, config = _model_for_task("orientation", 4, "relative")
    assert isinstance(model, RelativeOrientationNet)
    checkpoint = make_checkpoint(type(model).__name__, {**config, "task": "orientation"}, model.state_dict())
    path = tmp_path / "relative.pt"
    save_checkpoint(path, checkpoint)

    evaluated = _load_model("orientation", path)
    vsn = VirtualSensorNetwork.from_checkpoints(orientation_path=path)
    assert isinstance(evaluated, RelativeOrientationNet)
    assert isinstance(vsn.orientation, RelativeOrientationNet)
    assert torch.equal(evaluated(_valid_encoded_mask()), vsn.orientation(_valid_encoded_mask()))


def test_historical_orientation_checkpoint_still_loads_without_model_type(tmp_path: Path):
    legacy = OrientationNet(base=4)
    checkpoint = make_checkpoint(
        type(legacy).__name__,
        {"task": "orientation", "in_channels": 1, "angles": list(legacy.angles), "base": 4},
        legacy.state_dict(),
    )
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, checkpoint)
    assert isinstance(_load_model("orientation", path), OrientationNet)
    assert isinstance(VirtualSensorNetwork.from_checkpoints(orientation_path=path).orientation, OrientationNet)
