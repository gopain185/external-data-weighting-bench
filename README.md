# external-data-weighting-bench

How much weight should a large external training pool get when the target data are few and the pool comes from a
shifted distribution? Can cross-validation choose that weight safely? This repository simulates the question with a
known truth. Classes are Gaussian, and the pool moves every class mean by a chosen fraction of the typical distance
between two classes. The pool is 14 times the in-domain set, the test set is in-domain, and the model is a weighted
multinomial logistic regression.

It came out of a text-classification competition, where a public pool of passages 14 times the size of the train
set helped an n-gram SVM most at a pool share of 0.85, and less at both 0.5 and 0.95.

## Results

Gain in balanced accuracy of a fixed pool share `lam` over the best in-domain-only model (10 replicates, 20,000 test
rows each):

| in-domain rows | shift | lam 0.25 | lam 0.5 | lam 0.7 | lam 0.85 | lam 0.95 | best lam |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 150 | 0 | +0.032 | +0.071 | +0.093 | +0.101 | +0.103 | 0.95 |
| 150 | 0.5 | +0.023 | +0.045 | +0.046 | +0.036 | +0.024 | 0.7 |
| 600 | 0.25 | +0.011 | +0.018 | +0.018 | +0.013 | +0.009 | 0.7 |
| 600 | 0.5 | +0.006 | +0.001 | -0.013 | -0.030 | -0.042 | 0.25 |
| 2,700 | 0.1 | +0.003 | +0.006 | +0.006 | +0.006 | +0.005 | 0.85 |
| 2,700 | 0.25 | +0.002 | -0.000 | -0.006 | -0.012 | -0.017 | 0.25 |
| 2,700 | 1.0 | -0.018 | -0.060 | -0.101 | -0.136 | -0.164 | 0 |

Each way of choosing the weight, against the best fixed weight (worst condition of the 15, 5 replicates each):

| rule | worst regret | at |
| --- | --- | --- |
| 5-fold CV on in-domain rows, pool always in training | -0.008 | 150 rows, shift 1.0 |
| two-stage: pool model, then in-domain fit pulled towards it | -0.015 | 150 rows, shift 0.5 |
| random folds over in-domain and pool rows together | -0.173 | 2,700 rows, shift 1.0 |
| concatenate, every row the same weight | -0.168 | 2,700 rows, shift 1.0 |
| no pool (ridge strength by CV) | -0.102 | 150 rows, shift 0 |

A pool with a skewed class prior (Dirichlet(1)), 600 rows, shift 0.25: raw row weights keep +0.002 of the +0.019
gain. Class-balanced weights, where every class's pool rows get the same total, keep +0.017. 20% random pool labels
cost only 0.001.

The full write-up is in `kaggle/notebook_findings.md`.

## Use

```
pip install -r requirements.txt
python -m pytest                 # 13 tests
python -m xdwbench results       # all three experiments, about 20 minutes on a laptop CPU
python kaggle/build_kaggle.py --reuse-results   # Kaggle dataset and notebook folders
```

- `xdwbench/world.py`: the generator (class means, pool shift, pool prior, label noise).
- `xdwbench/model.py`: weighted multinomial logistic regression with an L2 pull towards a prior weight matrix.
- `xdwbench/methods.py`: pool weighting (class-balanced or raw), the two-stage fit, and the selection rules.
- `xdwbench/experiments.py`: the experiments and their summaries.

Kaggle notebook: https://www.kaggle.com/code/locdunhat/how-much-weight-should-external-data-get
Kaggle dataset: https://www.kaggle.com/datasets/locdunhat/external-data-weighting-simulations

## Limitations

The model is linear, the classes are Gaussian and every class shifts by the same amount. The direction of the
effects is the point, not their size. A neural model fine-tuned in two stages can behave differently from the linear
two-stage fit here.

## License

MIT for the code, CC0 for the generated data.
