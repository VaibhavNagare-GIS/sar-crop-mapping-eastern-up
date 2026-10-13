"""Run: python tests/selftest_phase6d.py -> 'SELFTEST OK'. Extra training data in cross-validation, on synthetic features."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import p6_extra_train as et

rng = np.random.default_rng(0)
centres = {1: np.zeros(6), 2: np.r_[1.2, 1.2, 0, 0, 0, 0], 6: np.r_[0, 0, 1.5, 1.5, 0, 0]}
def draw(n, noise):
    y = rng.choice([1, 2, 6], n, p=[0.5, 0.3, 0.2])
    return np.stack([centres[c] for c in y]) + rng.normal(0, noise, (n, 6)), y

X, y = draw(600, 2.0)
g = rng.integers(0, 30, len(y))
Xe, ye = draw(900, 2.0)
# extra data with all classes: more samples from the same distribution must not hurt
full = et.run(X, y, g, Xe, ye, seeds=1, trees=30, leaf=3)
assert set(full) == {"base", "far", "far_w0.3"} and full["far"]["f1"] > full["base"]["f1"] - 0.05, full
# extra data without one class (as the far set has no Sugarcane): the folds must still work; scores are valid but the class balance shifts
part = et.run(X, y, g, Xe[ye != 6], ye[ye != 6], seeds=1, trees=30, leaf=3)
assert all(0.3 < v["f1"] <= 1 for v in part.values()), part
P = et.cv_with_extra(X, y, g, None, None, None, 0, 20, 3)
assert P.shape == (600, 3) and np.allclose(P.sum(1), 1)
print("SELFTEST OK")
