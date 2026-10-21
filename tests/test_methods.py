import numpy as np

from xdwbench import methods as m
from xdwbench.world import World


def test_weights_give_the_pool_its_share():
    y_in = np.repeat(np.arange(3), [10, 10, 10])
    y_pool = np.repeat(np.arange(3), [100, 10, 1])
    for balance in (True, False):
        w = m.weights(y_in, y_pool, 3, 0.7, balance)
        assert np.isclose(w[30:].sum(), 0.7) and np.isclose(w[:30].sum(), 0.3)
    w = m.weights(y_in, y_pool, 3, 0.7, True)[30:]
    per_class = np.bincount(y_pool, weights=w)
    assert np.allclose(per_class, 0.7 / 3)  # balanced: every class's pool rows carry the same total


def test_cv_takes_the_pool_when_it_is_clean_and_drops_it_when_it_is_far_off():
    rng = np.random.default_rng(0)
    for shift, expect in [(0.0, lambda lam: lam >= 0.7), (3.0, lambda lam: lam <= 0.25)]:
        inst = World(shift=shift).draw(rng)
        X_in, y_in = inst.domain(150, rng)
        X_pool, y_pool = inst.pool(2100, rng)
        lam, scores = m.choose_lambda_cv(X_in, y_in, X_pool, y_pool, 10, rng)
        assert expect(lam), (shift, lam, scores)


def test_folds_are_stratified():
    y = np.repeat(np.arange(4), 10)
    f = m.folds(y, 5, np.random.default_rng(0))
    for k in range(5):
        assert np.bincount(y[f == k], minlength=4).tolist() == [2, 2, 2, 2]
