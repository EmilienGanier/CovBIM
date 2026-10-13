# test_core.py
# import _core
from CovBIM import the_tester


def test_covbim():
    input_value = 5
    result = the_tester(input_value)
    assert result is not None
    assert result == 2*input_value