import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    n, d = x.shape
    X = np.ones(n).reshape(-1, 1)
    for i in range(1, degree+1):
        x_pol = x ** i
        X = np.hstack((X,x_pol))
    
    weights = np.linalg.pinv(X.T @ X) @ X.T @ y

    return weights.flatten().tolist()





