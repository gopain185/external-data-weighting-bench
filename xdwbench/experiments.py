"""The three experiments. Every function returns a long table; the summarise_* functions reduce it.

Within one replicate all methods see the same world, the same in-domain rows, the same pool and the same test set,
so differences between methods are paired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import methods as m
from .model import fit, predict
from .world import World, balanced_accuracy

SEED = 20261022
N_TEST = 20000
POOL_RATIO = 14  # pool rows per in-domain row (the Name That LLM pool is about 14 times its train set)
N_IN = (150, 600, 2700)
SHIFTS = (0.0, 0.1, 0.25, 0.5, 1.0)
L2_GRID = (1e-3, 1e-2, 1e-1)


def _data(world: World, n_in: int, rng):
    inst = world.draw(rng)
    X_in, y_in = inst.domain(n_in, rng)
    X_pool, y_pool = inst.pool(POOL_RATIO * n_in, rng)
    X_te, y_te = inst.domain(N_TEST, rng)
    return X_in, y_in, X_pool, y_pool, X_te, y_te


def curves(reps: int = 10, n_in=N_IN, shifts=SHIFTS) -> pd.DataFrame:
    """Test balanced accuracy of every fixed setting: pool weight lam, two-stage pull tau, in-domain-only l2."""
    rows = []
    for n in n_in:
        for shift in shifts:
            for r in range(reps):
                rng = np.random.default_rng([SEED, 1, n, int(shift * 100), r])
                w = World(shift=shift)
                X_in, y_in, X_pool, y_pool, X_te, y_te = _data(w, n, rng)
                K = w.K
                score = lambda model: balanced_accuracy(y_te, predict(model, X_te))
                base = dict(n_in=n, shift=shift, rep=r)
                for lam in m.LAMBDAS:
                    rows.append(dict(base, method="mixture", param=lam,
                                     acc=score(m.mixture(X_in, y_in, X_pool, y_pool, K, lam))))
                prior = m.pool_model(X_pool, y_pool, K)
                rows.append(dict(base, method="pool_only", param=1.0, acc=score(prior)))
                for tau in m.TAUS:
                    rows.append(dict(base, method="two_stage", param=tau, acc=score(m.two_stage(X_in, y_in, prior, K, tau))))
                for l2 in L2_GRID:
                    rows.append(dict(base, method="in_only", param=l2, acc=score(fit(X_in, y_in, K, l2=l2))))
    return pd.DataFrame(rows)


def summarise_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy of each pool weight minus the best in-domain-only model, and the best lam per condition."""
    best_in = df[df.method == "in_only"].groupby(["n_in", "shift", "rep"]).acc.max().rename("in_best")
    mix = df[df.method == "mixture"].join(best_in, on=["n_in", "shift", "rep"])
    mix["gain"] = mix.acc - mix.in_best
    t = mix.pivot_table(index=["n_in", "shift"], columns="param", values="gain", aggfunc="mean")
    t.columns = [f"lam={c:g}" for c in t.columns]
    t["best_lam"] = mix.groupby(["n_in", "shift", "param"]).gain.mean().unstack().idxmax(axis=1)
    t["in_only"] = best_in.groupby(["n_in", "shift"]).mean()
    return t.reset_index()


def selection(reps: int = 5, n_in=N_IN, shifts=SHIFTS) -> pd.DataFrame:
    """Each way of choosing the pool weight, scored on the in-domain test set, next to the oracle choice."""
    rows = []
    for n in n_in:
        for shift in shifts:
            for r in range(reps):
                rng = np.random.default_rng([SEED, 2, n, int(shift * 100), r])
                w = World(shift=shift)
                X_in, y_in, X_pool, y_pool, X_te, y_te = _data(w, n, rng)
                K = w.K
                score = lambda model: balanced_accuracy(y_te, predict(model, X_te))
                fixed = {lam: score(m.mixture(X_in, y_in, X_pool, y_pool, K, lam)) for lam in m.LAMBDAS}
                lam_cv, cv_scores = m.choose_lambda_cv(X_in, y_in, X_pool, y_pool, K, rng)
                lam_mixed, _ = m.choose_lambda_mixed(X_in, y_in, X_pool, y_pool, K, rng)
                prior = m.pool_model(X_pool, y_pool, K)
                tau_cv, _ = m.choose_tau_cv(X_in, y_in, prior, K, rng)
                f = m.folds(y_in, 5, rng)
                l2_cv = max(L2_GRID, key=lambda l2: np.mean([
                    balanced_accuracy(y_in[f == k], predict(fit(X_in[f != k], y_in[f != k], K, l2=l2), X_in[f == k]))
                    for k in range(5)]))
                concat = len(y_pool) / (len(y_pool) + len(y_in))
                base = dict(n_in=n, shift=shift, rep=r, oracle_acc=max(fixed.values()))
                for name, param, acc in [
                    ("in_only_cv", l2_cv, score(fit(X_in, y_in, K, l2=l2_cv))),
                    ("concat", concat, score(m.mixture(X_in, y_in, X_pool, y_pool, K, concat, balance=False))),
                    ("lam_in_domain_cv", lam_cv, fixed[lam_cv]),
                    ("lam_mixed_cv", lam_mixed, fixed[lam_mixed]),
                    ("lam_oracle", max(fixed, key=fixed.get), max(fixed.values())),
                    ("two_stage_cv", tau_cv, score(m.two_stage(X_in, y_in, prior, K, tau_cv))),
                ]:
                    rows.append(dict(base, method=name, param=param, acc=acc,
                                     cv_estimate=cv_scores[lam_cv] if name == "lam_in_domain_cv" else np.nan))
    return pd.DataFrame(rows)


def summarise_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Mean regret against the oracle pool weight (negative = worse), per condition and selection rule."""
    d = df.assign(regret=df.acc - df.oracle_acc)
    t = d.pivot_table(index=["n_in", "shift"], columns="method", values="regret", aggfunc="mean")
    cv = d[d.method == "lam_in_domain_cv"]
    t["cv_optimism"] = (cv.cv_estimate - cv.acc).groupby([cv.n_in, cv["shift"]]).mean()
    t["oracle_acc"] = d.groupby(["n_in", "shift"]).oracle_acc.mean()
    return t.reset_index()


def pool_quality(reps: int = 10, n: int = 600, shift: float = 0.25) -> pd.DataFrame:
    """A skewed or noisy pool: class-balanced against raw pool weights, and the lam that in-domain CV picks."""
    rows = []
    for alpha in (np.inf, 1.0):
        for noise in (0.0, 0.2):
            for r in range(reps):
                rng = np.random.default_rng([SEED, 3, int(np.isinf(alpha)), int(noise * 100), r])
                w = World(shift=shift, pool_alpha=alpha, label_noise=noise)
                X_in, y_in, X_pool, y_pool, X_te, y_te = _data(w, n, rng)
                K = w.K
                in_best = max(balanced_accuracy(y_te, predict(fit(X_in, y_in, K, l2=l2), X_te)) for l2 in L2_GRID)
                for balance in (True, False):
                    fixed = {lam: balanced_accuracy(y_te, predict(
                        m.mixture(X_in, y_in, X_pool, y_pool, K, lam, balance), X_te)) for lam in m.LAMBDAS}
                    lam_cv, _ = m.choose_lambda_cv(X_in, y_in, X_pool, y_pool, K, rng, balance)
                    rows.append(dict(pool_prior="balanced" if np.isinf(alpha) else "skewed (Dirichlet 1)",
                                     label_noise=noise, rep=r, weighting="class-balanced" if balance else "raw",
                                     in_only=in_best, best_lam=max(fixed, key=fixed.get), best_acc=max(fixed.values()),
                                     lam_cv=lam_cv, cv_acc=fixed[lam_cv]))
    return pd.DataFrame(rows)


def summarise_pool_quality(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(best_gain=df.best_acc - df.in_only, cv_gain=df.cv_acc - df.in_only)
    return (d.groupby(["pool_prior", "label_noise", "weighting"])
            .agg(in_only=("in_only", "mean"), best_gain=("best_gain", "mean"), cv_gain=("cv_gain", "mean"),
                 best_lam=("best_lam", "median"), lam_cv=("lam_cv", "median"))
            .reset_index())
