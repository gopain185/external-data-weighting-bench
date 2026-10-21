# How Much Weight Should External Data Get?

A common setup: a small labelled set from the target distribution, and an external pool many times larger that was
collected somewhere else. In *Name That LLM* the pool is public passages from the same 18 models, written to other
prompts under other settings. You can train on both, but how much weight should the pool get? Is that choice safe to
make with cross-validation?

This notebook simulates that setup with a known truth. Classes are Gaussian, and the pool moves every class mean by
a chosen fraction (`shift`) of the typical distance between two classes. The pool is 14 times the in-domain set, and
the test set is in-domain. The classifier is a weighted multinomial logistic regression. `lam` is the pool's share of
the total training weight, split over classes in the in-domain proportions.

It measures:

1. test accuracy against `lam`, for 150, 600 and 2,700 in-domain rows and five shift levels;
2. how close each way of choosing `lam` gets to the best fixed value: CV on in-domain rows only, CV on random folds
   of everything, plain concatenation, and a two-stage fit (pool first, then in-domain rows with a pull towards the
   pool model);
3. what a skewed class prior or wrong labels in the pool change, with class-balanced and raw pool weights.

Everything runs on CPU in about 20 to 30 minutes. The package source is embedded in the next cell, so the notebook
is self-contained.
