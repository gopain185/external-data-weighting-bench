import numpy as np
from scipy.optimize import check_grad

from xdwbench import model
from xdwbench.world import World, balanced_accuracy


def test_fit_separates_easy_classes():
    inst = World(sep=8.0).draw(np.random.default_rng(0))
    X, y = inst.domain(500, np.random.default_rng(1))
    W = model.fit(X, y, 10)
    assert balanced_accuracy(y, model.predict(W, X)) > 0.95


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X, y, w = rng.normal(size=(40, 3)), rng.integers(0, 4, 40), rng.random(40)
    prior = (rng.normal(size=(3, 4)), None)
    captured = {}

    def fake_minimize(f, theta0, jac, method, options):
        captured["f"] = f
        class R:
            x = theta0
        return R()

    orig = model.minimize
    model.minimize = fake_minimize
    try:
        model.fit(X, y, 4, sample_weight=w, l2=0.3, prior=prior)
    finally:
        model.minimize = orig
    f = captured["f"]
    theta = rng.normal(size=3 * 4 + 4)
    assert check_grad(lambda t: f(t)[0], lambda t: f(t)[1], theta) < 1e-5


def test_strong_pull_returns_the_prior():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(50, 3)), rng.integers(0, 4, 50)
    P = rng.normal(size=(3, 4))
    W, _ = model.fit(X, y, 4, l2=1e6, prior=(P, None))
    assert np.allclose(W, P, atol=1e-3)


def test_sample_weights_equal_duplication():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(30, 2)), rng.integers(0, 3, 30)
    w = np.ones(30)
    w[:10] = 2.0
    a = model.fit(X, y, 3, sample_weight=w, l2=0.1)
    b = model.fit(np.concatenate([X, X[:10]]), np.concatenate([y, y[:10]]), 3, l2=0.1)
    assert np.allclose(a[0], b[0], atol=1e-4)
