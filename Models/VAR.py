from statsmodels.tsa.api import VAR
import numpy as np


def get_var_coefficients(data, p):
    var = VAR(data)
    model_fitted = var.fit(p)
    return np.hstack(model_fitted.coefs)


def predict(coefficients, data):
    m = coefficients.shape[0]
    p = int(coefficients.shape[1]/m)
    y = []
    for i in range(len(data)-p):
        X = data[i:i+p,:]
        X = X[::-1].reshape(-1,1)
        y.append(np.dot(coefficients, X))
    # Convert to numpy array and make it 2d
    return np.array(y).reshape(-1,m)