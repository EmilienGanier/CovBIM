# test_core.py
# import _core
from CovBIM import _core


def test_the_tester_binding():
    input_value = 5
    result = _core.the_tester(input_value)
    assert result is not None
    assert result == 2*input_value