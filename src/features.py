"""Feature helpers shared by all modelling scripts (Sentinel-1 tables and NDVI tables)."""
import numpy as np
from scipy.ndimage import uniform_filter1d

import study

KINDS = ("raw", "smooth", "summary", "both")
MODELS = ("rf", "et", "hgb", "lr")


def band_cols():
    n = study.N_BINS
    return [f"VV_{i:02d}" for i in range(n)], [f"VH_{i:02d}" for i in range(n)]


def ndvi_cols():
    return [f"NDVI_{i:02d}" for i in range(study.N_BINS)]


def col_groups(df):
    """Column groups present in a table: Sentinel-1 bands and/or NDVI bands."""
    g = []
    if "VV_00" in df:
        g.append(sum(band_cols(), []))
    if "NDVI_00" in df:
        g.append(ndvi_cols())
    if not g:
        raise ValueError("table has neither VV_00 nor NDVI_00")
    return g


def drop_empty(df):
    """Drop pixels where ANY sensor has no valid bin at all. Returns (df, number dropped). Callers must print it."""
    empty = np.zeros(len(df), bool)
    for cols in col_groups(df):
        empty |= df[cols].isna().all(axis=1).to_numpy()
    return df[~empty].reset_index(drop=True), int(empty.sum())


def fill_time(a):
    """(rows, bins) with NaN gaps -> linear interpolation over time, nearest value at the ends."""
    a = np.array(a, dtype=float)
    bad = np.isnan(a)
    if bad.all(axis=1).any():
        raise ValueError(f"{int(bad.all(axis=1).sum())} rows have no valid bins; call drop_empty first")
    x = np.arange(a.shape[1])
    for r in np.nonzero(bad.any(axis=1))[0]:
        a[r, bad[r]] = np.interp(x[bad[r]], x[~bad[r]], a[r, ~bad[r]])
    return a


def series(df):
    """name -> (rows, bins) gap-filled arrays: VV, VH, VHVV (dB) if Sentinel-1 is present, NDVI if present."""
    out = {}
    if "VV_00" in df:
        vv_c, vh_c = band_cols()
        vv, vh = fill_time(df[vv_c].to_numpy()), fill_time(df[vh_c].to_numpy())
        out.update(VV=vv, VH=vh, VHVV=vh - vv)
    if "NDVI_00" in df:
        out["NDVI"] = fill_time(df[ndvi_cols()].to_numpy())
    if not out:
        raise ValueError("table has neither VV_00 nor NDVI_00")
    return out


def _stats(a):
    """11 summary numbers per row of a (rows, bins): level, spread, extremes, timing, early/mid/late means."""
    e, m, l = [a[:, i].mean(1) for i in np.array_split(np.arange(a.shape[1]), 3)]
    cols = [a.mean(1), a.std(1), a.min(1), a.max(1), a.max(1) - a.min(1),
            a.argmax(1).astype(float), a.argmin(1).astype(float), e, m, l, l - e]
    return np.column_stack(cols), ["mean", "std", "min", "max", "range", "argmax", "argmin", "early", "mid", "late", "late_minus_early"]


def make_X(df, kind="raw"):
    """Model matrix and names. kind: raw bins | smooth (3-bin moving mean) | summary (11 stats per series) | both (raw + summary)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    blocks, names = [], []
    for k, a in series(df).items():
        if kind == "smooth":
            a = uniform_filter1d(a, 3, axis=1, mode="nearest")
        if kind != "summary":
            blocks.append(a)
            names += [f"{k}_{i:02d}" for i in range(a.shape[1])]
        if kind in ("summary", "both"):
            st, nm = _stats(a)
            blocks.append(st)
            names += [f"{k}_{x}" for x in nm]
    return np.hstack(blocks), names


def make_model(name="rf", leaf=1, trees=200, seed=0):
    """Classifiers used in the sweeps. All reweight classes; larger `leaf` = smoother (safer on noisy labels)."""
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "rf":
        return RandomForestClassifier(trees, min_samples_leaf=leaf, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "et":
        return ExtraTreesClassifier(trees, min_samples_leaf=leaf, class_weight="balanced", n_jobs=-1, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=max(leaf, 20),
                                              l2_regularization=1.0, class_weight="balanced", random_state=seed)
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))
    raise ValueError(f"model must be one of {MODELS}")


def cv_predict(X, y, groups, seed=0, trees=200, leaf=1, model="rf", proba=False):
    """Out-of-fold predictions (or class probabilities) with 5-fold CV that keeps each group (spatial block) together."""
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(make_model(model, leaf, trees, seed), X, y, groups=groups, cv=cv,
                             method="predict_proba" if proba else "predict")
