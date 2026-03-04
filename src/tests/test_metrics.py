import numpy as np

from src.metrics import (
    statistical_parity_difference,
    equal_opportunity_difference,
)


def test_spd_on_toy_example():
    # a=1 group has 2/3 positives, a=0 group has 1/3 positives => SPD = 2/3 - 1/3 = 1/3
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    a = np.array([1, 1, 1, 0, 0, 0])

    spd = statistical_parity_difference(y_pred, a)
    assert abs(spd - (1.0 / 3.0)) < 1e-9


def test_eod_on_toy_example():
    # Construct y_true/y_pred so that:
    # For a=1: TPR = 2/2 = 1.0
    # For a=0: TPR = 1/2 = 0.5
    # EOD = 1.0 - 0.5 = 0.5
    y_true = np.array([1, 1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    a = np.array([1, 1, 1, 0, 0, 0])

    eod = equal_opportunity_difference(y_true, y_pred, a)
    assert abs(eod - 0.5) < 1e-9