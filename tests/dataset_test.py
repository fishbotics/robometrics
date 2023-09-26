import pytest

from robometrics.datasets import demo, motion_benchmaker, mpinets

try:
    import geometrout
except ImportError:
    pytest.skip("geometrout not available", allow_module_level=True)


def test_datasets():
    # Try loading datasets in robometrics
    for k in [demo, mpinets, motion_benchmaker]:
        data = k()
        assert len(data[list(data.keys())[0]]) > 0
