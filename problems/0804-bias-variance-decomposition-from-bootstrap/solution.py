import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.array(predictions)
    mean_predictions = np.mean(predictions, axis = 0).reshape(-1, 1)
    y_true = np.array(y_true).reshape(-1, 1)
    bias_squared = np.mean((mean_predictions - y_true)**2)
    variance =  np.mean(np.mean((predictions.T - mean_predictions)**2, axis = 1)) 
    mse = np.mean(np.mean((predictions.T - y_true)**2, axis = 1)) 


    return {
        'bias_squared': bias_squared, 
        'variance': variance, 
        'mse': mse
        }
