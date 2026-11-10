import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    X_train = np.array(X_train) 
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    n_train_samples, n_features = X_train.shape
    n_val_samples, n_features = X_val.shape 

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)


    train_errors = []
    val_errors = []
    diagnosis = ""

    for n in train_sizes:
        X_t = np.ones(n).reshape(-1, 1)
        X_v = np.ones(n_val_samples).reshape(-1, 1)
        for d in range(1, degree+1):
            X_train_poly = X_train[:n, :] ** d 
            X_val_poly = X_val ** d 

            X_t = np.hstack((X_t, X_train_poly))
            X_v = np.hstack((X_v, X_val_poly))
        y_t = y_train[:n]
        weights = np.linalg.pinv(X_t) @ y_t

        y_train_pred = X_t @ weights
        y_val_pred = X_v @ weights 

        error_train = y_train_pred - y_t
        error_val = y_val_pred - y_val


        mse_train = np.mean(error_train**2)
        mse_val =  np.mean(error_val**2)
        
        train_errors.append(float(mse_train))
        val_errors.append(float(mse_val))

        if mse_train > bias_threshold:
            diagnosis = 'high_bias'
        elif (mse_val - mse_train) > variance_threshold:
            diagnosis = 'high_variance'
        else:
            diagnosis = 'good_fit'


    return {
        'train_errors': train_errors, 
        'val_errors': val_errors, 
        'diagnosis': diagnosis}
    







