import numpy as np


def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error regression loss"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def true_sign(y_true, y_pred):
    """Compute the percentage of correct sign predictions"""
    total = y_true.shape[0] if len(y_true.shape) == 1 else y_true.shape[0]*y_true.shape[1]
    return ((y_true>0) == (y_pred>0)).sum()/total


def matrix_rmse(true_matrix, estimated_matrix):
    """Compute the root mean squared error between two matrices, element wise and lag wise"""
    m = true_matrix.shape[0]
    if true_matrix.shape[1] > estimated_matrix.shape[1]:
        estimated_matrix = np.hstack((estimated_matrix, np.zeros((estimated_matrix.shape[0], true_matrix.shape[1] - estimated_matrix.shape[1]))))
    elif true_matrix.shape[1] < estimated_matrix.shape[1]:
        true_matrix = np.hstack((true_matrix, np.zeros((true_matrix.shape[0], estimated_matrix.shape[1] - true_matrix.shape[1]))))
    metrics = {'all': root_mean_squared_error(true_matrix, estimated_matrix), 'sign': true_sign(true_matrix, estimated_matrix)}
    # Variable wise
    for i in range(estimated_matrix.shape[0]):
        metrics[f'x_{i}'] = root_mean_squared_error(true_matrix[i,:], estimated_matrix[i, :])
        metrics[f'x_{i}_sign'] = true_sign(true_matrix[i,:], estimated_matrix[i, :])
    # Lag wise
    for i in range(0, estimated_matrix.shape[1], m):
        metrics[f'lag_{int(i/m)}'] = root_mean_squared_error(true_matrix[:,i:i+m], estimated_matrix[:,i:i+m])
        metrics[f'lag_{int(i/m)}_sign'] = true_sign(true_matrix[:,i:i+m], estimated_matrix[:,i:i+m])
    return metrics
