from sfn.envs import AssetRegistry


def test_asset_registry_discovers_shapes():
    reg = AssetRegistry()
    shapes = reg.list_shapes()
    assert "square-triangle" in shapes
    assets = reg.get("square-triangle")
    assert assets.base_urdf.exists()
    assert assets.peg_test_urdf.exists()
