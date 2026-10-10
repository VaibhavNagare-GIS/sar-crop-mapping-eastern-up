"""Phase 2b: which season do the labels fit best? TRAIN fields only; the test split is never touched.

Usage: python p2_season_test.py data/phase2/2020-10-01_pilot data/phase2/2021-10-01_pilot
Prints grouped-cross-validated macro-F1 for each folder. Evidence, not proof: crop rotation repeats across years.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

import features


def score(df):
    df, dead = features.drop_empty(df)
    X, _ = features.make_X(df)
    df["pred"] = features.cv_predict(X, df.label, df.block)
    fld = df.groupby("field").agg(label=("label", "first"), pred=("pred", lambda s: s.mode().iloc[0]))
    return (f1_score(df.label, df.pred, average="macro"), f1_score(fld.label, fld.pred, average="macro"),
            len(df), df.field.nunique(), dead)


def main():
    print("folder | pixel macro-F1 | field macro-F1 | pixels | fields | dead pixels")
    for d in sys.argv[1:]:
        df = pd.read_csv(Path(d) / "features.csv", dtype={"chip": str})
        if not (df.split == "train").all():
            raise SystemExit(f"{d} contains test fields. Season test must use train fields only.")
        p, f, n, nf, dead = score(df)
        print(f"{Path(d).name} | {p:.3f} | {f:.3f} | {n} | {nf} | {dead}")


if __name__ == "__main__":
    main()
