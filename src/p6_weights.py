"""Phase 6b: class weights on the predicted probabilities to raise macro-F1 (decision thresholds). TRAIN data only.
python src/p6_weights.py SPEC --level field|pixel [--leaf L] [--classes 3]
Out-of-fold probabilities come from block-wise CV. Weights are tuned on one random half of the spatial blocks and scored on the other half,
both ways and over several seeds, so the reported gain is not measured on the data used for tuning.
Rule: adopt only if the mean held-out gain in macro-F1 is 0.02 or more. Writes data/phase3/weights_<level>.json."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import features
import p3_sweep as sw
import p4_final as p4

GRID = np.exp(np.linspace(np.log(0.5), np.log(4.0), 9))


def tune(P, y, classes):
    """Weights (first class fixed at 1) that maximise macro-F1 of argmax(P * w)."""
    best, bw = -1.0, None
    for w in itertools.product(GRID, repeat=len(classes) - 1):
        w = np.array((1.0, *w))
        f = f1_score(y, classes[(P * w).argmax(1)], average="macro")
        if f > best:
            best, bw = f, w
    return bw, best


def holdout_gain(P, y, g, classes, seeds=5):
    gains = []
    blocks = np.unique(g)
    for s in range(seeds):
        order = np.random.default_rng(s).permutation(blocks)
        halves = [np.isin(g, order[: len(order) // 2]), np.isin(g, order[len(order) // 2:])]
        for a, b in ((0, 1), (1, 0)):
            w, _ = tune(P[halves[a]], y[halves[a]], classes)
            gains.append(f1_score(y[halves[b]], classes[(P[halves[b]] * w).argmax(1)], average="macro") - f1_score(y[halves[b]], classes[P[halves[b]].argmax(1)], average="macro"))
    return float(np.mean(gains)), float(np.std(gains))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--level", choices=["field", "pixel"], required=True)
    ap.add_argument("--leaf", type=int, default=10)
    ap.add_argument("--classes", default="3")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    parts, meta = sw.load(a.spec)
    parts, meta = p4.prep(parts, meta, a.classes)
    X, y, g = sw.build(parts, meta, a.level, "raw")
    P = features.cv_predict(X, y, g, seed=0, trees=100, leaf=a.leaf, proba=True)
    classes = np.unique(y)
    w, f_in = tune(P, y, classes)
    base = f1_score(y, classes[P.argmax(1)], average="macro")
    gain, sd = holdout_gain(P, y, g, classes)
    print(f"{a.level}: F1 without weights {base:.3f}; held-out gain from tuned weights {gain:+.3f} (sd {sd:.3f}) -> {'ADOPT' if gain >= 0.02 else 'no'}")
    print(f"weights for classes {classes.tolist()} tuned on all out-of-fold data: {np.round(w, 2).tolist()} (in-sample F1 {f_in:.3f})")
    out = Path(a.data) / "phase3" / f"weights_{a.level}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(spec=a.spec, classes=a.classes, leaf=a.leaf, labels=classes.tolist(), weights=w.tolist(), holdout_gain=gain, base_f1=base), indent=2))


if __name__ == "__main__":
    main()
