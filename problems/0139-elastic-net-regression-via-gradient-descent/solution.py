import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n_sample, n_feature = X.shape 
    weights = np.zeros(n_feature).reshape(n_feature, -1)
    bias = 0.0
    y = y.reshape(-1, 1)
    for i in range(max_iter):
        y_pred = X @ weights + bias 
        error = y_pred - y 

        grad_w = (1/n_sample) * X.T @ error + alpha1 * np.sign(weights) + 2 * alpha2 * weights
        grad_b = (1/n_sample) * np.sum(error) 
        weights_old = weights.copy()
        weights = weights_old - learning_rate * grad_w
        bias = bias - learning_rate * grad_b

        if np.sum(np.abs(grad_w)) < tol:
            break 
    return (weights.flatten(), bias)









