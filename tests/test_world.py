import numpy as np

from xdwbench.world import World, balanced_accuracy


def test_domain_is_balanced_and_shapes():
    inst = World().draw(np.random.default_rng(0))
    X, y = inst.domain(100, np.random.default_rng(1))
    assert X.shape == (100, 20)
    assert np.bincount(y).tolist() == [10] * 10


def test_shift_moves_pool_means_by_the_stated_amount():
    w = World(shift=0.5)
    inst = w.draw(np.random.default_rng(0))
    dist = np.linalg.norm(inst.mu_pool - inst.mu, axis=1)
    assert np.allclose(dist, 0.5 * w.sep)


def test_mean_separation_matches_sep():
    w = World(K=200, d=50, sep=2.0)
    mu = w.draw(np.random.default_rng(0)).mu
    pair = np.linalg.norm(mu[:100] - mu[100:], axis=1).mean()
    assert abs(pair - 2.0) < 0.1


def test_label_noise_and_skewed_prior():
    rng = np.random.default_rng(0)
    clean = World(pool_alpha=0.5).draw(rng)
    _, y = clean.pool(20000, rng)
    freq = np.bincount(y, minlength=10) / len(y)
    assert np.allclose(freq, clean.pool_prior, atol=0.02)
    noisy = World(label_noise=1.0).draw(rng)
    X, y = noisy.pool(20000, rng)
    nearest = np.linalg.norm(X[:, None, :] - noisy.mu_pool[None], axis=2).argmin(1)
    assert np.mean(nearest == y) < 0.2  # labels carry almost no information


def test_balanced_accuracy():
    y = np.array([0, 0, 0, 1])
    assert balanced_accuracy(y, np.array([0, 0, 0, 0])) == 0.5
