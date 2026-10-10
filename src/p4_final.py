"""Phase 4: freeze the configuration, then the ONE test evaluation.
  python p4_final.py freeze --field-spec A+D --pixel-spec A2+D2 --classes 3 --leaf-field 10 --leaf-pixel 10 [--bench NAME=SPEC ...] [--embed CSV]
  python p4_final.py test [--again] [--boot 2000]
freeze writes results/FROZEN_CONFIG.json (settings chosen on TRAIN data only). test reads it, trains on all train fields, predicts the test
fields once and writes results/. A lock file stops accidental second looks at the test set."""
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

import features
import p3_sweep as sw

FALLOW = sw.FALLOW
NAMES = {1: "Wheat", 2: "Mustard", 3: "Lentil", 6: "Sugarcane"}


def commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "no-git"


def relabel(meta, cs):
    """Class scheme: '3' merges Mustard+Lentil (label 2), '4' keeps four crops, '5' keeps Fallow. Returns row mask and labels."""
    m = (meta.label != FALLOW).to_numpy() if cs in ("3", "4") else np.ones(len(meta), bool)
    lab = meta.label.copy()
    if cs == "3":
        lab = lab.where(~lab.isin((2, 3)), 2)
    return m, lab


def prep(parts, meta, cs):
    m, lab = relabel(meta, cs)
    meta = meta.assign(label=lab)
    return [p[m] for p in parts], meta[m]


def check_disjoint(tr, te):
    if set(tr.field) & set(te.field):
        raise SystemExit("LEAK: some fields are in both train and test")
    if set(tr.block) & set(te.block):
        raise SystemExit("LEAK: some spatial blocks are in both train and test")


def core(y, P, classes):
    pred = classes[P.argmax(1)]
    cm = confusion_matrix(y, pred, labels=classes)
    row, col = cm.sum(axis=1), cm.sum(0)
    pa = np.divide(np.diag(cm), row, out=np.zeros(len(classes)), where=row > 0)
    ua = np.divide(np.diag(cm), col, out=np.zeros(len(classes)), where=col > 0)
    f1 = np.divide(2 * pa * ua, pa + ua, out=np.zeros(len(classes)), where=(pa + ua) > 0)
    aucs = [roc_auc_score(y == c, P[:, i]) if 0 < (y == c).sum() < len(y) else np.nan for i, c in enumerate(classes)]
    return dict(f1=float(f1.mean()), bal=float(pa.mean()), oa=float(np.trace(cm) / cm.sum()), auc=float(np.nanmean(aucs)),
                cm=cm, pa=pa, ua=ua, f1c=f1, aucc=aucs)


def boot(units, y, probs, classes, draws):
    """Cluster bootstrap: resample whole units (fields). probs = {model: P}. -> {model: (B,4)} of f1, bal, oa, auc."""
    order = np.argsort(units, kind="stable")
    cuts = np.flatnonzero(np.diff(units[order])) + 1
    groups = np.split(order, cuts)
    out = {k: np.empty((len(draws), 4)) for k in probs}
    for b, d in enumerate(draws):
        idx = np.concatenate([groups[i] for i in d])
        for k, P in probs.items():
            r = core(y[idx], P[idx], classes)
            out[k][b] = (r["f1"], r["bal"], r["oa"], r["auc"])
    return out


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def fit(parts_tr, meta_tr, parts_te, meta_te, level, cfg, leaf, extra=None):
    """Train on all train rows, predict test rows once. -> y_test, P_test, classes, unit ids (fields), model."""
    Xtr, ytr, _ = sw.build(parts_tr, meta_tr, level, "raw")
    Xte, yte, _ = sw.build(parts_te, meta_te, level, "raw")
    if extra is not None:
        Xtr, Xte = np.hstack([Xtr, extra[0]]), np.hstack([Xte, extra[1]])
    model = features.make_model("rf", leaf, cfg["trees"], cfg["seed"]).fit(Xtr, ytr)
    units = meta_te.field.to_numpy() if level == "pixel" else np.sort(meta_te.field.unique())
    return yte, model.predict_proba(Xte), model.classes_, units, model


def field_mean_table(table, meta):
    return table.reindex(meta.index).groupby(meta.field.to_numpy()).mean().to_numpy()


def test(cfg, out_dir, data_dir, boot_n, again):
    out_dir, lock = Path(out_dir), Path(out_dir) / "TEST_USED.lock"
    if lock.exists() and lock.read_text().startswith("DONE") and not again:
        raise SystemExit(f"{lock} exists: the test set was already used. Rerun with --again only if you accept that the results are then no longer pre-registered.")
    if lock.exists() and again:
        print("WARNING: second look at the test set; the results are no longer pre-registered.")
    out_dir.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"STARTED {dt.datetime.now().isoformat(timespec='seconds')}\n")
    cs, rng = cfg["classes"], np.random.default_rng(cfg["seed"])
    draws_cache, res, rows_pc, cms, fpreds = {}, {}, [], {}, {}

    def fld(spec, extra_tables=()):
        parts, meta = sw.load(spec, ("train", "test"))
        parts, meta = prep(parts, meta, cs)
        tr, te = (meta.split == "train").to_numpy(), (meta.split == "test").to_numpy()
        check_disjoint(meta[tr], meta[te])
        ex = None
        if extra_tables:
            ex = tuple(np.hstack([field_mean_table(t, meta[s]) for t in extra_tables]) for s in (tr, te))
        return [p[tr] for p in parts], meta[tr], [p[te] for p in parts], meta[te], ex

    # field level: primary + benchmarks, all on the same test fields
    models = {"primary": (cfg["field_spec"], None)}
    models.update({k: (v, None) for k, v in cfg.get("bench", {}).items()})
    if cfg.get("embed"):
        models["s1+embedding"] = (cfg["field_spec"], pd.read_csv(cfg["embed"], index_col="pid"))
    got = {}
    for name, (spec, emb) in models.items():
        ptr, mtr, pte, mte, ex = fld(spec, (emb,) if emb is not None else ())
        y, P, classes, units, m = fit(ptr, mtr, pte, mte, "field", cfg, cfg["leaf_field"], ex)
        got[name] = pd.DataFrame(P, index=units, columns=[f"p{c}" for c in classes]).assign(y=y)
        if name == "primary":
            base_cls = int(mtr.groupby("field").label.first().mode()[0])        # biggest class in TRAIN fields
            import joblib
            Path(data_dir, "phase4").mkdir(parents=True, exist_ok=True)
            joblib.dump(m, Path(data_dir) / "phase4" / "model_field.joblib", compress=3)
            classes_ref = classes
    common = sorted(set.intersection(*[set(d.index) for d in got.values()]))
    if len(common) != len(got["primary"]):
        print(f"NOTE: {len(got['primary']) - len(common)} test fields are missing from some benchmark and are left out of ALL field-level numbers")
    ref = got["primary"].loc[common]
    yf = ref.y.to_numpy()
    probs = {n: d.loc[common][[f"p{c}" for c in classes_ref]].to_numpy() for n, d in got.items()}
    draws = rng.integers(0, len(common), (boot_n, len(common)))
    bs = boot(np.arange(len(common)), yf, probs, classes_ref, draws)
    res["field"], diffs = {}, {}
    for n, P in probs.items():
        r = core(yf, P, classes_ref)
        res["field"][n] = dict(**{k: r[k] for k in ("f1", "bal", "oa", "auc")}, ci={k: ci(bs[n][:, i]) for i, k in enumerate(("f1", "bal", "oa", "auc"))},
                               n_fields=len(common), baseline_oa=float((yf == base_cls).mean()))
        cms[("field", n)] = pd.DataFrame(r["cm"], index=[NAMES.get(c, c) for c in classes_ref], columns=[NAMES.get(c, c) for c in classes_ref])
        for i, c in enumerate(classes_ref):
            rows_pc.append(dict(level="field", model=n, cls=NAMES.get(c, c), PA=r["pa"][i], UA=r["ua"][i], F1=r["f1c"][i], AUC=r["aucc"][i], n=int(r["cm"][i].sum())))
        if n != "primary":
            d = bs["primary"] - bs[n]
            diffs[n] = {k: dict(diff=res["field"]["primary"][k] - res["field"][n][k], ci=ci(d[:, i]), p_better=float((d[:, i] > 0).mean()))
                        for i, k in enumerate(("f1", "bal", "oa", "auc"))}
    res["field_diff_primary_minus"] = diffs
    ref.assign(pred=classes_ref[probs["primary"].argmax(1)]).to_csv(out_dir / "test_predictions_field.csv", index_label="field")

    # pixel level (the map model) and its aggregation to fields
    parts, meta = sw.load(cfg["pixel_spec"], ("train", "test"))
    parts, meta = prep(parts, meta, cs)
    tr, te = (meta.split == "train").to_numpy(), (meta.split == "test").to_numpy()
    check_disjoint(meta[tr], meta[te])
    y, P, classes, units, m = fit([p[tr] for p in parts], meta[tr], [p[te] for p in parts], meta[te], "pixel", cfg, cfg["leaf_pixel"])
    pdraw = rng.integers(0, len(np.unique(units)), (boot_n, len(np.unique(units))))
    bp = boot(pd.factorize(units)[0], y, {"pixel": P}, classes, pdraw)["pixel"]
    r = core(y, P, classes)
    res["pixel"] = dict(**{k: r[k] for k in ("f1", "bal", "oa", "auc")}, ci={k: ci(bp[:, i]) for i, k in enumerate(("f1", "bal", "oa", "auc"))},
                        n_pixels=len(y), n_fields=len(np.unique(units)), baseline_oa=float((y == int(meta[tr].label.mode()[0])).mean()))
    cms[("pixel", "pixel")] = pd.DataFrame(r["cm"], index=[NAMES.get(c, c) for c in classes], columns=[NAMES.get(c, c) for c in classes])
    for i, c in enumerate(classes):
        rows_pc.append(dict(level="pixel", model="pixel", cls=NAMES.get(c, c), PA=r["pa"][i], UA=r["ua"][i], F1=r["f1c"][i], AUC=r["aucc"][i], n=int(r["cm"][i].sum())))
    pf = pd.DataFrame(P, columns=[f"p{c}" for c in classes]).assign(field=units, y=y).groupby("field").agg({**{f"p{c}": "mean" for c in classes}, "y": "first"})
    ra = core(pf.y.to_numpy(), pf[[f"p{c}" for c in classes]].to_numpy(), classes)
    res["pixel_model_field_vote"] = {k: ra[k] for k in ("f1", "bal", "oa", "auc")}
    cms[("pixel", "field_vote")] = pd.DataFrame(ra["cm"], index=[NAMES.get(c, c) for c in classes], columns=[NAMES.get(c, c) for c in classes])

    out_dir.mkdir(parents=True, exist_ok=True)
    for (lvl, n), df in cms.items():
        df.to_csv(out_dir / f"confusion_{lvl}_{n}.csv")
    pd.DataFrame(rows_pc).to_csv(out_dir / "per_class.csv", index=False)
    res["config"] = cfg
    (out_dir / "test_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    import joblib
    joblib.dump(m, out_dir / "model.joblib", compress=3)
    mb = (out_dir / "model.joblib").stat().st_size / 1e6
    print(f"pixel model saved: {mb:.0f} MB" + ("  (over 50 MB: do NOT commit it)" if mb > 50 else ""))
    lock.write_text(f"DONE {dt.datetime.now().isoformat(timespec='seconds')}\n")
    for lvl in ("field", "pixel"):
        items = res[lvl].items() if lvl == "field" else [("pixel", res[lvl])]
        for n, r in items:
            print(f"{lvl:5s} {n:14s} F1 {r['f1']:.3f} {r['ci']['f1']}  bal {r['bal']:.3f}  OA {r['oa']:.3f} (baseline {r['baseline_oa']:.3f})  AUC {r['auc']:.3f} {r['ci']['auc']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--field-spec", required=True)
    f.add_argument("--pixel-spec", required=True)
    f.add_argument("--classes", default="3", choices=["3", "4", "5"])
    f.add_argument("--leaf-field", type=int, default=10)
    f.add_argument("--leaf-pixel", type=int, default=10)
    f.add_argument("--trees", type=int, default=500)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--bench", nargs="*", default=[], help="NAME=SPEC field-level benchmarks, e.g. s1_s2=A+D+S2")
    f.add_argument("--embed", default="")
    f.add_argument("--out", default="results")
    t = sub.add_parser("test")
    t.add_argument("--again", action="store_true")
    t.add_argument("--boot", type=int, default=2000)
    t.add_argument("--out", default="results")
    t.add_argument("--data", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "freeze":
        out.mkdir(parents=True, exist_ok=True)
        cfg = dict(field_spec=a.field_spec, pixel_spec=a.pixel_spec, classes=a.classes, leaf_field=a.leaf_field, leaf_pixel=a.leaf_pixel,
                   trees=a.trees, seed=a.seed, bench=dict(b.split("=", 1) for b in a.bench), embed=a.embed,
                   frozen=dt.datetime.now().isoformat(timespec="seconds"), git_commit=commit())
        p = out / "FROZEN_CONFIG.json"
        if p.exists():
            raise SystemExit(f"{p} exists. Delete it by hand only if the frozen configuration really has to change, and state that in the report.")
        p.write_text(json.dumps(cfg, indent=2))
        print(f"wrote {p}\n{json.dumps(cfg, indent=2)}")
    else:
        cfg = json.loads((out / "FROZEN_CONFIG.json").read_text())
        test(cfg, out, a.data, a.boot, a.again)


if __name__ == "__main__":
    main()
