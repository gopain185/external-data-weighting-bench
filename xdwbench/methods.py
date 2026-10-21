"""Ways to combine the in-domain rows with the pool, and ways to choose how much weight the pool gets.

`lam` is the pool's share of the total training weight. With `balance=True` the pool's weight is split over classes
in the in-domain proportions (each class's pool rows together get lam of that class's weight), which undoes a skewed
pool prior; with `balance=False` every pool row gets the same weight.
"""

from __future__ import annotations

import numpy as np

from .model import fit, predict
from .world import balanced_accuracy

LAMBDAS = (0.0, 0.25, 0.5, 0.7, 0.85, 0.95)
TAUS = (1e-3, 1e-2, 1e-1, 1.0)  # strength of the pull towards the pool model in two_stage
L2 = 1e-2


def weights(y_in, y_pool, K, lam, balance):
    """Row weights for np.concatenate([in-domain rows, pool rows])."""
    w_in = np.full(len(y_in), (1 - lam) / len(y_in))
    if lam == 0:
        return np.concatenate([w_in, np.zeros(len(y_pool))])
    if balance:
        share = np.bincount(y_in, minlength=K) / len(y_in)
        n_pool = np.maximum(np.bincount(y_pool, minlength=K), 1)
        w_pool = lam * share[y_pool] / n_pool[y_pool]
        w_pool *= lam / w_pool.sum()  # classes absent from the pool: keep the pool's total at lam
    else:
        w_pool = np.full(len(y_pool), lam / len(y_pool))
    return np.concatenate([w_in, w_pool])


def mixture(X_in, y_in, X_pool, y_pool, K, lam, balance=True, l2=L2):
    if lam == 0:
        return fit(X_in, y_in, K, l2=l2)
    X = np.concatenate([X_in, X_pool])
    y = np.concatenate([y_in, y_pool])
    w = weights(y_in, y_pool, K, lam, balance)
    keep = w > 0
    return fit(X[keep], y[keep], K, sample_weight=w[keep], l2=l2)


def pool_model(X_pool, y_pool, K, balance=True):
    w = None
    if balance:
        n = np.maximum(np.bincount(y_pool, minlength=K), 1)
        w = 1.0 / n[y_pool]
    return fit(X_pool, y_pool, K, sample_weight=w, l2=L2)


def two_stage(X_in, y_in, prior, K, tau):
    """Fine-tune on the in-domain rows with an L2 pull of strength tau towards the pool model's weights."""
    return fit(X_in, y_in, K, l2=tau, prior=prior, init=prior)


def folds(y, n_folds, rng):
    """Stratified fold ids for the in-domain rows."""
    f = np.empty(len(y), int)
    for k in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == k))
        f[idx] = np.arange(len(idx)) % n_folds
    return f


def choose_lambda_cv(X_in, y_in, X_pool, y_pool, K, rng, balance=True, n_folds=5, lambdas=LAMBDAS):
    """In-domain CV: the pool is always in training, only in-domain rows are validated."""
    f = folds(y_in, n_folds, rng)
    score = {lam: [] for lam in lambdas}
    for k in range(n_folds):
        tr, va = f != k, f == k
        for lam in lambdas:
            m = mixture(X_in[tr], y_in[tr], X_pool, y_pool, K, lam, balance)
            score[lam].append(balanced_accuracy(y_in[va], predict(m, X_in[va])))
    means = {lam: float(np.mean(v)) for lam, v in score.items()}
    return max(means, key=means.get), means


def choose_lambda_mixed(X_in, y_in, X_pool, y_pool, K, rng, balance=True, n_folds=5, lambdas=LAMBDAS):
    """The tempting mistake: random folds over in-domain and pool rows together, so pool rows are validated too."""
    X = np.concatenate([X_in, X_pool])
    y = np.concatenate([y_in, y_pool])
    is_in = np.r_[np.ones(len(y_in), bool), np.zeros(len(y_pool), bool)]
    f = rng.permutation(np.arange(len(y)) % n_folds)
    score = {lam: [] for lam in lambdas}
    for k in range(n_folds):
        tr, va = f != k, f == k
        for lam in lambdas:
            m = mixture(X[tr & is_in], y[tr & is_in], X[tr & ~is_in], y[tr & ~is_in], K, lam, balance)
            score[lam].append(balanced_accuracy(y[va], predict(m, X[va])))
    means = {lam: float(np.mean(v)) for lam, v in score.items()}
    return max(means, key=means.get), means


def choose_tau_cv(X_in, y_in, prior, K, rng, n_folds=5, taus=TAUS):
    f = folds(y_in, n_folds, rng)
    score = {t: [] for t in taus}
    for k in range(n_folds):
        tr, va = f != k, f == k
        for t in taus:
            m = two_stage(X_in[tr], y_in[tr], prior, K, t)
            score[t].append(balanced_accuracy(y_in[va], predict(m, X_in[va])))
    means = {t: float(np.mean(v)) for t, v in score.items()}
    return max(means, key=means.get), means
