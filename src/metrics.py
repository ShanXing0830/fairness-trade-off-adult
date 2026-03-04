import numpy as np


def statistical_parity_difference(y_pred: np.ndarray, a: np.ndarray) -> float:
    """
    SPD = P(ŷ=1 | a=1) - P(ŷ=1 | a=0)
    a is a binary protected attribute (e.g., 1=Male, 0=Female)
    """
    y_pred = np.asarray(y_pred).astype(int).ravel()
    a = np.asarray(a).astype(int).ravel()

    p1 = y_pred[a == 1].mean()
    p0 = y_pred[a == 0].mean()
    return float(p1 - p0)


def true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    denom = (y_true == 1).sum()
    if denom == 0:
        return 0.0
    return float(((y_pred == 1) & (y_true == 1)).sum() / denom)


def equal_opportunity_difference(
    y_true: np.ndarray, y_pred: np.ndarray, a: np.ndarray
) -> float:
    """
    EOD = TPR(a=1) - TPR(a=0)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    a = np.asarray(a).astype(int).ravel()

    tpr1 = true_positive_rate(y_true[a == 1], y_pred[a == 1])
    tpr0 = true_positive_rate(y_true[a == 0], y_pred[a == 0])
    return float(tpr1 - tpr0)


def disparate_impact(y_pred: np.ndarray, a: np.ndarray, eps: float = 1e-12) -> float:
    """
    DI = P(ŷ=1 | a=0) / P(ŷ=1 | a=1)
    """
    y_pred = np.asarray(y_pred).astype(int).ravel()
    a = np.asarray(a).astype(int).ravel()

    p1 = y_pred[a == 1].mean()
    p0 = y_pred[a == 0].mean()
    return float(p0 / (p1 + eps))