from xdwbench import experiments as ex


def test_experiments_run_small():
    c = ex.curves(reps=1, n_in=(150,), shifts=(0.0, 1.0))
    assert set(c.method) == {"mixture", "pool_only", "two_stage", "in_only"}
    s = ex.summarise_curves(c)
    assert len(s) == 2 and "best_lam" in s
    sel = ex.selection(reps=1, n_in=(150,), shifts=(0.5,))
    assert (sel.acc <= sel.oracle_acc + 1e-12)[sel.method.str.startswith("lam_")].all()
    assert len(ex.summarise_selection(sel)) == 1
    q = ex.pool_quality(reps=1, n=150)
    assert len(ex.summarise_pool_quality(q)) == 8
