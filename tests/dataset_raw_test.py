from robometrics.datasets import demo_raw, motion_benchmaker_raw, mpinets_raw


def test_raw_dataset():
    # Try loading datasets in robometrics
    for k in [demo_raw, mpinets_raw, motion_benchmaker_raw]:
        data = k()
        assert len(data[list(data.keys())[0]]) > 0
