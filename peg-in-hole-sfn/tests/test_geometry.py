from sfn.geometry import decode_orientation, decode_position, encode_orientation, encode_position


def test_position_cardinal_signs():
    assert encode_position(0.001, 0.0) == (10, 9)
    assert encode_position(-0.001, 0.0) == (10, 11)
    assert encode_position(0.0, 0.001) == (11, 10)
    assert encode_position(0.0, -0.001) == (9, 10)


def test_position_round_trip():
    for row in range(21):
        for col in range(21):
            assert encode_position(*decode_position(row, col)) == (row, col)


def test_orientation_codec():
    idx = encode_orientation(3.1)
    assert decode_orientation(idx) in {2.0, 4.0}
