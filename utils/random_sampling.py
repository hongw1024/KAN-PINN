import numpy as np
# from pyDOE import lhs
from SALib.sample import sobol_sequence


def random_sampling(inputs, n, loc, dtype="float64"):
    """
    Inputs:    n = number of sampling
            type = ["ic", "bc", "f"]
    """
    if loc == "ic":
        x, t, u = inputs
        X = np.hstack((x.T, t.T))
        U = u.T
        idx = np.random.choice(X.shape[0], n, replace=False)
        return X[idx, :], U[idx, :]
    elif loc == "bc":
        x, t, u = inputs
        X = np.hstack((x, t))
        idx = np.random.choice(X.shape[0], n, replace=False)
        return X[idx, :], u[idx, :]
    elif loc == "f":
        lb, ub = inputs
        # x = lb + (ub - lb) * lhs(2, n)
        x = lb + (ub - lb) * sobol_sequence.sample(n, 2).astype(dtype)
        return x
    else:
        raise ValueError("input 'loc' can only be one of ['ic', 'bc', 'f']!")
