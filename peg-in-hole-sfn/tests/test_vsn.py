import pytest

torch = pytest.importorskip("torch")
from sfn.models import VirtualSensorNetwork
from sfn.models.controllers import SFSSController


def test_vsn_mask_forward_shapes():
    vsn = VirtualSensorNetwork()
    mask = torch.zeros(2, 200, 250, dtype=torch.long)
    out = vsn(mask=mask)
    assert out.position_prob.shape == (2, 21, 21)
    assert out.orientation_prob.shape == (2, 11)


def test_empty_or_single_class_mask_is_invalid_and_holds_safely():
    vsn = VirtualSensorNetwork()
    for mask in (torch.zeros(1, 32, 32, dtype=torch.long), torch.ones(1, 32, 32, dtype=torch.long)):
        out = vsn(mask=mask)
        assert out.valid.tolist() == [False]
        assert out.invalid_reason == ("missing_peg_or_seam",)
        assert float(out.position_prob.detach().sum()) == 0.0
        assert float(out.orientation_prob.detach().sum()) == 0.0
        action = SFSSController(confidence_mode="ignore").act(out)
        assert action.physical.tolist() == [0.0, 0.0, 0.0]
        assert action.normalized.tolist() == [0.0, 0.0, 0.0]
