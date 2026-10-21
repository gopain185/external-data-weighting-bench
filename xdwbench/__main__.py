"""python -m xdwbench [outdir]: run every experiment, print the summaries and save the raw tables."""

import sys
import time
from pathlib import Path

from . import experiments as ex


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, fn, summary in [("exp1_curves", ex.curves, ex.summarise_curves),
                              ("exp2_selection", ex.selection, ex.summarise_selection),
                              ("exp3_pool_quality", ex.pool_quality, ex.summarise_pool_quality)]:
        t = time.time()
        df = fn()
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name} ({time.time() - t:.0f}s)\n{summary(df).round(4).to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
