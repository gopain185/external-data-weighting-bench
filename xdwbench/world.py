"""The simulated problem: a small labelled in-domain set, a large external pool from a shifted distribution.

Classes are Gaussian with unit covariance around means mu_k. The pool moves every class mean by `shift` times the
typical distance between two class means, each class in its own random direction (the external data were written
under different conditions, and each class reacts differently). The pool can also have uneven class frequencies and
a share of wrong labels. Test data are in-domain and balanced; the score is balanced accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class World:
    K: int = 10
    d: int = 20
    sep: float = 2.0  # typical distance between two class means, in noise standard deviations
    shift: float = 0.0  # pool mean displacement per class, as a fraction of sep
    pool_alpha: float = np.inf  # Dirichlet concentration of the pool's class frequencies (inf: balanced)
    label_noise: float = 0.0  # share of pool labels redrawn uniformly at random

    def draw(self, rng: np.random.Generator) -> "Instance":
        # Means with expected pairwise distance sep: each coordinate N(0, sep^2 / (2 d)).
        mu = rng.normal(0.0, self.sep / np.sqrt(2 * self.d), (self.K, self.d))
        u = rng.normal(size=(self.K, self.d))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        mu_pool = mu + self.shift * self.sep * u
        prior = (np.full(self.K, 1 / self.K) if np.isinf(self.pool_alpha)
                 else rng.dirichlet(np.full(self.K, self.pool_alpha)))
        return Instance(self, mu, mu_pool, prior)


@dataclass
class Instance:
    world: World
    mu: np.ndarray
    mu_pool: np.ndarray
    pool_prior: np.ndarray

    def domain(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """n in-domain rows with (as near as possible) equal class counts."""
        y = rng.permutation(np.arange(n) % self.world.K)
        return self.mu[y] + rng.normal(size=(n, self.world.d)), y

    def pool(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        y = rng.choice(self.world.K, size=n, p=self.pool_prior)
        x = self.mu_pool[y] + rng.normal(size=(n, self.world.d))
        flip = rng.random(n) < self.world.label_noise
        y = np.where(flip, rng.integers(0, self.world.K, n), y)
        return x, y


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))
