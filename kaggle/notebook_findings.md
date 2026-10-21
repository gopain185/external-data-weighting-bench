## Findings

Numbers are from the run committed with the package (`results/`). The notebook's own run uses the same seeds.

1. **The best pool weight falls as in-domain data grow and as the shift grows.** With 150 in-domain rows the pool
   helps at every shift: +0.10 at lam 0.95 with no shift, +0.046 at lam 0.7 with shift 0.5, +0.014 at lam 0.25
   with shift 1.0. With 2,700 rows
   the gain is never above +0.009. At shift 0.25 the best lam is 0.25, and lam 0.95 costs 0.017. At shift 0.5 the
   pool only hurts. The same pool can help a lot at one size and hurt at another.
2. **In-domain CV finds a good weight.** 5-fold CV on the in-domain rows, with the whole pool always in training,
   landed within 0.008 of the best fixed weight in every one of the 15 conditions, and within 0.003 in 11 of them.
   Its own score of the chosen weight was too high by at most 0.03, with the largest gaps at 150 rows.
3. **Validating on pool rows picks far too much pool.** Random folds over in-domain and pool rows together reward
   fitting the pool. That rule was as bad as plain concatenation (every row the same weight): at shift 0.5 and above it
   fell 0.02 to 0.17 below the best weight (0.05 to 0.17 with 600 rows or more), while in-domain CV lost at most
   0.008 there.
4. **A two-stage fit is close but not better.** Fitting on the pool and then on the in-domain rows with an L2 pull
   towards the pool model (strength chosen by in-domain CV) came within 0.002 of the best mixture with 2,700 rows,
   and lost 0.006 to 0.015 with 150 rows. In this linear setting, the pull and the mixture weight do the same job.
5. **The pool's class prior matters more than its label noise.** With 600 rows and shift 0.25, 20% random labels in
   the pool cut the gain from +0.019 to +0.018. A skewed class prior (Dirichlet(1)) with raw row weights cut it to
   +0.002. Class-balanced weights, where each class's pool rows get the same total, kept +0.017. Balancing the pool
   by class is cheap insurance.

For a competition with a small train set and a large external pool: choose the pool's weight with a CV that
validates on competition rows only, give each class's pool rows the same total weight, and choose the weight again
whenever the competition data or the pool change.

It is a stylised linear model with Gaussian classes, so the sizes are illustrative. The directions are the point.
