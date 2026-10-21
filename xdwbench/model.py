"""Weighted multinomial logistic regression with an L2 penalty towards a prior weight matrix.

With prior = 0 this is the usual ridge-penalised model; with prior = the weights of a model fitted on the pool it is
the linear analogue of pretraining on the pool and fine-tuning on the in-domain data with weight decay towards the
pretrained weights.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _unpack(theta, d, K):
    return theta[: d * K].reshape(d, K), theta[d * K:]


def fit(X, y, K, sample_weight=None, l2=1e-2, prior=None, init=None):
    """Returns (W, b) minimising sum_i w_i CE_i / sum_i w_i + l2 / 2 * ||W - prior||^2 (bias not penalised)."""
    n, d = X.shape
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    w = w / w.sum()
    P0 = np.zeros((d, K)) if prior is None else prior[0]
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1.0

    def f(theta):
        W, b = _unpack(theta, d, K)
        Z = X @ W + b
        Z -= Z.max(1, keepdims=True)
        logp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        loss = -(w * (Y * logp).sum(1)).sum() + 0.5 * l2 * ((W - P0) ** 2).sum()
        G = (np.exp(logp) - Y) * w[:, None]
        gW = X.T @ G + l2 * (W - P0)
        return loss, np.concatenate([gW.ravel(), G.sum(0)])

    theta0 = (np.concatenate([P0.ravel(), np.zeros(K)]) if init is None
              else np.concatenate([init[0].ravel(), init[1]]))
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    return _unpack(res.x, d, K)


def predict(model, X):
    W, b = model
    return (X @ W + b).argmax(1)
