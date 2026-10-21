"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public] [--notebook-only] [--reuse-results]

writes kaggle/dataset/ (one simulated problem per shift level, with in-domain, pool and test rows and their true
classes, the result tables of the three experiments and the datasheet) and kaggle/notebook/ (a CPU notebook that
embeds the package source and reruns the experiments).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xdwbench import experiments as ex  # noqa: E402
from xdwbench.world import World  # noqa: E402

OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/external-data-weighting-simulations"
NOTEBOOK_ID = "locdunhat/how-much-weight-should-external-data-get"
TITLE = "How Much Weight Should External Data Get?"


def build_dataset(public: bool, rerun: bool = True) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for shift in ex.SHIFTS:
        rng = np.random.default_rng([ex.SEED, 9, int(shift * 100)])
        inst = World(shift=shift).draw(rng)
        for split, (X, y) in [("in_domain", inst.domain(600, rng)), ("pool", inst.pool(ex.POOL_RATIO * 600, rng)),
                              ("test", inst.domain(5000, rng))]:
            df = pd.DataFrame(X.round(5), columns=[f"x{j}" for j in range(X.shape[1])])
            df.insert(0, "shift", shift)
            df.insert(1, "split", split)
            df.insert(2, "label", y)
            parts.append(df)
    pd.concat(parts).to_csv(out / "problems.csv", index=False)
    if rerun:
        for name, fn in [("exp1_curves", ex.curves), ("exp2_selection", ex.selection),
                         ("exp3_pool_quality", ex.pool_quality)]:
            fn().to_csv(out / f"{name}.csv", index=False)
    else:  # reuse the tables written by `python -m xdwbench results`
        for f in sorted((ROOT / "results").glob("exp*.csv")):
            (out / f.name).write_bytes(f.read_bytes())
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "External Data Weighting Simulations",
        "id": DATASET_ID,
        "subtitle": "Small in-domain sets, large shifted pools, known truth",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["benchmark", "classification", "tabular"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    files = sorted((ROOT / "xdwbench").glob("*.py"))
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in files}


SETUP = """

import os, sys
ROOT = "/kaggle/working/external-data-weighting-bench" if os.path.isdir("/kaggle/working") else os.path.abspath("external-data-weighting-bench")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
"""

IMPORTS = """import time, warnings
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from xdwbench import experiments as ex
pd.set_option("display.precision", 4); pd.set_option("display.width", 160)
"""

EXP1 = """t = time.time(); e1 = ex.curves(); print(round(time.time() - t), "s")
s1 = ex.summarise_curves(e1); s1
"""

PLOT1 = """fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
lam_cols = [c for c in s1.columns if c.startswith("lam=")]
lams = [float(c[4:]) for c in lam_cols]
for ax, (n, g) in zip(axes, s1.groupby("n_in")):
    for _, row in g.iterrows():
        ax.plot(lams, row[lam_cols].to_numpy(float), marker="o", label=f"shift {row['shift']:g}")
    ax.axhline(0, color="k", lw=0.8); ax.set_title(f"{n} in-domain rows, pool 14x")
    ax.set_xlabel("pool share of the training weight")
axes[0].set_ylabel("balanced accuracy minus best in-domain-only model"); axes[-1].legend()
plt.tight_layout(); plt.show()
"""

EXP2 = """t = time.time(); e2 = ex.selection(); print(round(time.time() - t), "s")
ex.summarise_selection(e2)
"""

EXP3 = """t = time.time(); e3 = ex.pool_quality(); print(round(time.time() - t), "s")
ex.summarise_pool_quality(e3)
"""

SAMPLE = """import glob
from xdwbench import methods as m
from xdwbench.model import predict
from xdwbench.world import balanced_accuracy
hits = glob.glob("/kaggle/input/**/problems.csv", recursive=True)
if hits:
    df = pd.read_csv(hits[0]); rows = []
    for shift, g in df.groupby("shift"):
        part = {s: (d.filter(like="x").to_numpy(), d.label.to_numpy()) for s, d in g.groupby("split")}
        (Xi, yi), (Xp, yp), (Xt, yt) = part["in_domain"], part["pool"], part["test"]
        lam_cv, _ = m.choose_lambda_cv(Xi, yi, Xp, yp, 10, np.random.default_rng(0))
        rows.append({"shift": shift, "lam_cv": lam_cv, **{f"lam={lam:g}": balanced_accuracy(yt, predict(m.mixture(Xi, yi, Xp, yp, 10, lam), Xt)) for lam in m.LAMBDAS}})
    display(pd.DataFrame(rows))
else:
    print("dataset not attached")
"""


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + SETUP),
        ("code", IMPORTS),
        ("markdown", "## 1. Accuracy against the pool's weight\n\nFor every in-domain size and shift, the gain in "
                     "balanced accuracy of a fixed pool share `lam` over the best in-domain-only model (the best of "
                     "three ridge strengths, chosen on the test set, so the comparison is strict). 10 replicates per "
                     "cell, 20,000 test rows each."),
        ("code", EXP1),
        ("code", PLOT1),
        ("markdown", "## 2. Choosing the weight\n\nEach rule's test accuracy minus that of the best fixed `lam` "
                     "(the oracle, chosen on the test set). `lam_in_domain_cv`: 5-fold CV on the in-domain rows with "
                     "the whole pool always in training. `lam_mixed_cv`: random folds over in-domain and pool rows "
                     "together. `concat`: every row the same weight. `two_stage_cv`: fit on the pool, then on the "
                     "in-domain rows with an L2 pull towards the pool model, its strength chosen by in-domain CV. "
                     "`in_only_cv`: no pool, ridge strength by CV. `cv_optimism`: the CV score of the chosen `lam` "
                     "minus its test score. 5 replicates per cell."),
        ("code", EXP2),
        ("markdown", "## 3. A skewed or noisy pool\n\n600 in-domain rows, shift 0.25. The pool's class frequencies "
                     "are either equal or drawn from a Dirichlet(1), and 0% or 20% of its labels are random. "
                     "`class-balanced` gives every class's pool rows the same total weight; `raw` gives every pool "
                     "row the same weight. Gains are over the best in-domain-only model, for the oracle `lam` and "
                     "for the `lam` chosen by in-domain CV. 10 replicates per cell."),
        ("code", EXP3),
        ("markdown", "## 4. The problems in the dataset\n\nThe companion dataset holds one simulated problem per "
                     "shift level (600 in-domain rows, 8,400 pool rows, 5,000 test rows, true classes included)."),
        ("code", SAMPLE),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (kind, src) in enumerate(_cells()):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "external-data-weighting.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "external-data-weighting.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
        "keywords": ["benchmark", "tabular"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public, rerun="--reuse-results" not in sys.argv))
    print("wrote", build_notebook(public))
