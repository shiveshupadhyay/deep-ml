import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = X.shape
    feature_indices = range(n_features)
    poly_columns = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(feature_indices, d):
            if len(combo) == 0:
                term = np.ones((n_samples, 1))
            else:
                term = np.prod(X[:, combo], axis=1, keepdims=True)
            poly_columns.append(term)
    X_poly = np.hstack(poly_columns)
    X_poly_sorted = np.sort(X_poly, axis=1)
    
    return X_poly_sorted
