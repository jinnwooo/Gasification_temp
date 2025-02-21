# Metric을 정의하는 코드
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-7):
    """
    y_true, y_pred: shape (N,) 일 때 동작.
    여러 타겟 각각을 분리해서 사용할 수 있음.
    """
    mask = (np.abs(y_true) > epsilon)
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-9))) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # 1D 배열이면 차원 추가
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
        y_pred = y_pred[:, np.newaxis]

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    non_zero_mask = denominator != 0  # 0으로 나누는 것 방지

    smape_values = np.zeros(y_true.shape[1])  # 타겟별 SMAPE 저장
    for i in range(y_true.shape[1]):
        if np.any(non_zero_mask[:, i]):  # 모든 값이 0이면 0 반환
            smape_values[i] = np.mean(
                (np.abs(y_true[:, i] - y_pred[:, i]) / denominator[:, i])[non_zero_mask[:, i]]) * 100

    return smape_values