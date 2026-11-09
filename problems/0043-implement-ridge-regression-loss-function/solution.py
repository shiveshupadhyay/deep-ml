import numpy as np

def ridge_loss(X: np.ndarray, 
	w: np.ndarray, y_true: np.ndarray, 
	alpha: float) -> float:
	w = w.reshape(-1, 1)
	y_true = y_true.reshape(-1, 1)
	y_pred = X @ w 
	loss = np.mean((y_pred - y_true) ** 2) + alpha * np.sum(w ** 2) 
	return loss 
	
